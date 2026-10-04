"""experiments.py - định nghĩa các thí nghiệm B/T/F, quy tắc chọn cấu hình (chỉ dùng VAL), và bộ chạy hàng loạt.

Mọi quyết định được ghi vào `<out_dir>/decisions.json` để kiểm toán được; các quy tắc chọn được ĐẶT TRƯỚC khi
xem kết quả (viết ở đây, không chỉnh theo kết quả):

  Backbone (Bước 1)   : ứng viên = các backbone có macro-F1 val >= (tốt nhất - 0.003); chọn cái có độ trễ batch-1
                        sơ bộ thấp nhất trong số đó ("đủ tốt trong độ trễ đo được", slide trang 33).
                        Backbone "nhẹ" = macro-F1 cao nhất trong các mạng <= 1 GMAC (dùng cho ensemble và biểu đồ
                        đánh đổi).
  Nhiễu (Bước 2)      : T00 chạy 3 seed; std_T00 = std mẫu (ddof=1) của macro-F1 val.
  Một yếu tố thắng    : Δ = macro-F1 val (yếu tố) - macro-F1 val (T00, seed 0) >= max(2 * std_T00, 0.003).
  Kết hợp (T20)       : ghép yếu tố thắng tốt nhất của mỗi "ô" (slot) không xung đột. Trục G (độ phân giải, số
                        epoch) chỉ được phân tích, KHÔNG ghép vào công thức cuối để giữ cùng ngân sách tính toán
                        với mốc T00 (so sánh I2 công bằng) và đủ 3 seed trong hạn mức GPU.
  Công thức cuối      : trong {T00, các yếu tố thắng, T20} lấy macro-F1 val cao nhất; nếu nhiều ứng viên chênh
                        nhau chưa tới ngưỡng thì lấy cái ít thay đổi nhất.

Dòng lệnh (chạy từ thư mục code/, các đường dẫn truyền qua tuỳ chọn chung):
    python experiments.py list
    python experiments.py backbones [--only B01,B02]
    python experiments.py select-backbone
    python experiments.py noise
    python experiments.py training [--include-extra] [--only T03,T07]
    python experiments.py winners
    python experiments.py combo
    python experiments.py recipe
    python experiments.py final-train
Các lệnh đều idempotent: lần chạy đã có summary.json sẽ được bỏ qua, phiên bị ngắt thì chạy lại là tiếp tục.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as TR  # noqa: E402

BACKBONE_RUNS = {
    "B01": "resnet50", "B02": "resnext50", "B03": "convnext_tiny", "B04": "deit_small",
    "B05": "swin_tiny", "B06": "efficientnet_b0", "B07": "mobilenetv3",
}
BACKBONE_FAMILY = {"resnet50": "ResNet", "resnext50": "ResNeXt", "convnext_tiny": "ConvNeXt",
                   "deit_small": "Transformer (DeiT)", "swin_tiny": "Transformer (Swin)",
                   "efficientnet_b0": "Nhẹ (EfficientNet)", "mobilenetv3": "Nhẹ (MobileNetV3)"}

BACKBONE_MARGIN = 0.003
MIN_DELTA = 0.003
NOISE_K = 2.0
LIGHT_GMAC = 1.0
FINAL_SEEDS = (0, 1, 2)

# Trục A-G của GUIDE mục 3. `slot`: các yếu tố cùng ô loại trừ nhau khi kết hợp. `adoptable=False`: chỉ phân tích.
TRAIN_EXPERIMENTS = [
    dict(exp_id="T01", axis="A", desc="frozen", slot="init", core=True, adoptable=True,
         changes="init=frozen: đóng băng backbone, chỉ train head", overrides=dict(init="frozen")),
    dict(exp_id="T02", axis="A", desc="scratch", slot="init", core=True, adoptable=True,
         changes="init=scratch (không tiền huấn luyện); LR toàn mạng 1e-3 vì 1e-4 là LR của fine-tune",
         overrides=dict(init="scratch", lr_backbone=1e-3, lr_head=1e-3)),
    dict(exp_id="T03", axis="B", desc="color", slot="aug", core=True, adoptable=True,
         changes="aug=color (+ ColorJitter)", overrides=dict(aug="color")),
    dict(exp_id="T04", axis="B", desc="trivial", slot="aug", core=True, adoptable=True,
         changes="aug=trivial (+ TrivialAugmentWide)", overrides=dict(aug="trivial")),
    dict(exp_id="T05", axis="B", desc="cutmix", slot="mix", core=True, adoptable=True,
         changes="mix=cutmix, alpha=1, xác suất 0,5 mỗi batch", overrides=dict(mix="cutmix", mix_prob=0.5)),
    dict(exp_id="T06", axis="B", desc="mixup", slot="mix", core=True, adoptable=True,
         changes="mix=mixup, alpha=1, xác suất 0,5 mỗi batch", overrides=dict(mix="mixup", mix_prob=0.5)),
    dict(exp_id="T07", axis="C", desc="labelsmooth", slot="loss", core=True, adoptable=True,
         changes="loss=label smoothing eps=0,1", overrides=dict(loss="ls", label_smoothing=0.1)),
    dict(exp_id="T08", axis="C", desc="focal", slot="loss", core=True, adoptable=True,
         changes="loss=focal gamma=2", overrides=dict(loss="focal", focal_gamma=2.0)),
    dict(exp_id="T09", axis="C", desc="weightedce", slot="loss", core=True, adoptable=True,
         changes="loss=CE có trọng số 1/n_c (chuẩn hoá trung bình 1)",
         overrides=dict(loss="ce_weighted", class_weight_beta=0.0)),
    dict(exp_id="T10", axis="D", desc="balanced", slot="sampler", core=True, adoptable=True,
         changes="sampler cân bằng lớp (oversample)", overrides=dict(sampler="balanced")),
    dict(exp_id="T11", axis="E", desc="lr_equal", slot="lr", core=True, adoptable=True,
         changes="LR head = LR backbone = 1e-4 (không gấp 10 lần)", overrides=dict(lr_head=1e-4)),
    dict(exp_id="T12", axis="F", desc="ema", slot="ema", core=True, adoptable=True,
         changes="EMA trọng số, decay 0,998", overrides=dict(ema_decay=0.998)),
    dict(exp_id="T13", axis="G", desc="res256", slot="res", core=True, adoptable=False,
         changes="độ phân giải huấn luyện 256 thay vì 224", overrides=dict(img_size=256)),
    # --- mở rộng (chạy khi còn ngân sách GPU): python experiments.py training --include-extra ---
    dict(exp_id="T14", axis="B", desc="geo", slot="aug", core=False, adoptable=True,
         changes="aug=geo (+ lật dọc)", overrides=dict(aug="geo")),
    dict(exp_id="T15", axis="B", desc="mildcrop", slot="aug", core=False, adoptable=True,
         changes="aug=mildcrop (RandomResizedCrop scale 0,35-1)", overrides=dict(aug="mildcrop")),
    dict(exp_id="T16", axis="E", desc="sgd", slot="optim", core=False, adoptable=True,
         changes="SGD momentum 0,9 (lr 1e-2/1e-1, wd 5e-4) thay AdamW",
         overrides=dict(optimizer="sgd", lr_backbone=1e-2, lr_head=1e-1, weight_decay=5e-4)),
    dict(exp_id="T17", axis="E", desc="nowarmup", slot="warmup", core=False, adoptable=True,
         changes="không warmup", overrides=dict(warmup_epochs=0.0)),
    dict(exp_id="T18", axis="F", desc="droppath", slot="droppath", core=False, adoptable=True,
         changes="stochastic depth 0,1", overrides=dict(drop_path_rate=0.1)),
    dict(exp_id="T19", axis="G", desc="ep20", slot="epochs", core=False, adoptable=False,
         changes="20 epoch thay vì 12", overrides=dict(epochs=20)),
]
CONFLICTS = [("T09", "T10")]  # CE có trọng số + sampler cân bằng = cân bằng hai lần


# --------------------------------------------------------------------------- #
# Đường dẫn và dựng Config
# --------------------------------------------------------------------------- #
PATHS = dict(images_dir="data/images", labels_dir="data/labels", out_dir="runs",
             pred_dir="predictions", curves_dir="curves")
GLOBAL = dict(epochs=None, num_workers=2, cache_images=False)  # ghi đè toàn cục (hạ ngân sách thì ghi vào báo cáo)


def set_paths(**kw) -> None:
    for k, v in kw.items():
        if k in PATHS and v is not None:
            PATHS[k] = str(v)
        elif k in GLOBAL and v is not None:
            GLOBAL[k] = v


def make_cfg(exp_id: str, **overrides) -> TR.Config:
    kw = dict(PATHS)
    kw.update({k: v for k, v in GLOBAL.items() if v is not None})
    kw.update(overrides)
    return TR.Config(exp_id=exp_id, **kw)


def decisions_path() -> Path:
    return Path(PATHS["out_dir"]) / "decisions.json"


def load_decisions() -> dict:
    p = decisions_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_decisions(d: dict) -> None:
    p = decisions_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def update_decisions(**kw) -> dict:
    d = load_decisions()
    d.update(kw)
    save_decisions(d)
    return d


def load_summary(exp_id: str, seed: int = 0) -> dict | None:
    p = Path(PATHS["out_dir"]) / exp_id / f"seed{seed}" / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_summaries() -> pd.DataFrame:
    rows = []
    for p in sorted(Path(PATHS["out_dir"]).glob("*/seed*/summary.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        rows.append({k: v for k, v in s.items() if k != "config"} | {"cfg": s["config"]})
    return pd.DataFrame(rows)


def _need_gpu_warning() -> None:
    import torch
    if not torch.cuda.is_available():
        print("CẢNH BÁO: không có GPU. Huấn luyện thật cần GPU (Kaggle/Colab); trên CPU sẽ rất chậm.")


# --------------------------------------------------------------------------- #
# Bước 1: backbone
# --------------------------------------------------------------------------- #
def run_backbones(only: list[str] | None = None, seed: int = 0) -> list[dict]:
    """B01-B07: cùng công thức nền T00, cùng seed, mỗi backbone một lần chạy."""
    _need_gpu_warning()
    out = []
    for exp_id, key in BACKBONE_RUNS.items():
        if only and exp_id not in only:
            continue
        cfg = make_cfg(exp_id, backbone=key, seed=seed, desc=key,
                       note=f"công thức nền T00, backbone {key}")
        out.append(TR.run(cfg))
    return out


def select_backbone(manual: str | None = None) -> dict:
    """Áp quy tắc chọn backbone (xem docstring module) và ghi vào decisions.json."""
    rows = {}
    for exp_id, key in BACKBONE_RUNS.items():
        s = load_summary(exp_id, 0)
        if s:
            rows[key] = s
    if len(rows) < 5:
        raise RuntimeError(f"mới có {len(rows)} backbone chạy xong (cần >= 5)")
    best_f1 = max(s["val_macro_f1"] for s in rows.values())
    cands = [k for k, s in rows.items() if s["val_macro_f1"] >= best_f1 - BACKBONE_MARGIN]
    lat = {k: (rows[k]["latency_b1_ms"] if rows[k]["latency_b1_ms"] is not None else float("inf")) for k in cands}
    primary = manual or min(cands, key=lambda k: (lat[k], -rows[k]["val_macro_f1"]))
    light_pool = [k for k, s in rows.items() if s["gmacs"] <= LIGHT_GMAC]
    light = max(light_pool, key=lambda k: rows[k]["val_macro_f1"]) if light_pool else None
    why = (f"ứng viên trong {BACKBONE_MARGIN} macro-F1 val của tốt nhất ({best_f1:.4f}): {cands}; "
           f"chọn độ trễ batch-1 thấp nhất" if not manual else "chọn thủ công")
    d = update_decisions(backbone_primary=primary, backbone_light=light, backbone_reason=why,
                         backbone_table={k: {"val_macro_f1": s["val_macro_f1"], "params_M": s["params_M"],
                                             "gmacs": s["gmacs"], "latency_b1_ms": s["latency_b1_ms"]}
                                         for k, s in rows.items()})
    print("Backbone chính:", primary, "| nhẹ:", light, "|", why)
    return d


def primary_exp_id() -> str:
    key = load_decisions()["backbone_primary"]
    return next(e for e, k in BACKBONE_RUNS.items() if k == key)


# --------------------------------------------------------------------------- #
# Bước 2: công thức huấn luyện
# --------------------------------------------------------------------------- #
def base_overrides() -> dict:
    return dict(backbone=load_decisions()["backbone_primary"])


def run_noise(seeds=FINAL_SEEDS) -> dict:
    """T00 = công thức nền trên backbone chính. Seed 0 dùng lại lần chạy B0x (cấu hình giống hệt); seed khác train
    mới. Tính std_T00 (nhiễu) từ macro-F1 val qua các seed."""
    _need_gpu_warning()
    d = load_decisions()
    if "backbone_primary" not in d:
        raise RuntimeError("chưa chọn backbone: chạy select-backbone trước")
    src = primary_exp_id()
    for sd in seeds:
        if sd == 0:
            TR.alias_run(src, "T00", 0, PATHS["out_dir"], PATHS["curves_dir"], desc=d["backbone_primary"],
                         note=f"công thức nền (= {src})")
        else:
            TR.run(make_cfg("T00", seed=sd, desc=d["backbone_primary"], note="công thức nền", **base_overrides()))
    f1 = [load_summary("T00", sd)["val_macro_f1"] for sd in seeds]
    std = float(np.std(f1, ddof=1)) if len(f1) > 1 else float("nan")
    update_decisions(noise={"seeds": list(seeds), "val_macro_f1": f1, "mean": float(np.mean(f1)), "std": std})
    print(f"T00 qua seed {list(seeds)}: macro-F1 val {f1} -> std = {std:.4f}")
    return load_decisions()["noise"]


def run_training(include_extra: bool = False, only: list[str] | None = None, seed: int = 0) -> list[dict]:
    """T01-T13 (+ T14-T19 nếu include_extra): mỗi lần chỉ khác T00 một yếu tố."""
    _need_gpu_warning()
    if "noise" not in load_decisions():
        raise RuntimeError("chưa có T00 nhiều seed: chạy noise trước")
    out = []
    for e in TRAIN_EXPERIMENTS:
        if only and e["exp_id"] not in only:
            continue
        if not only and not (e["core"] or include_extra):
            continue
        cfg = make_cfg(e["exp_id"], seed=seed, desc=e["desc"], note=e["changes"],
                       **{**base_overrides(), **e["overrides"]})
        out.append(TR.run(cfg))
    return out


def decide_winners() -> dict:
    """Δ so với T00 (seed 0); yếu tố thắng nếu Δ >= max(NOISE_K * std_T00, MIN_DELTA)."""
    d = load_decisions()
    base = load_summary("T00", 0)["val_macro_f1"]
    std = d["noise"]["std"]
    thr = max(NOISE_K * (0.0 if np.isnan(std) else std), MIN_DELTA)
    table, winners = {}, []
    for e in TRAIN_EXPERIMENTS:
        s = load_summary(e["exp_id"], 0)
        if not s:
            continue
        delta = s["val_macro_f1"] - base
        win = delta >= thr and e["adoptable"]
        table[e["exp_id"]] = {"desc": e["desc"], "axis": e["axis"], "slot": e["slot"],
                              "val_macro_f1": s["val_macro_f1"], "delta": delta, "winner": bool(win),
                              "adoptable": e["adoptable"]}
        if win:
            winners.append(e["exp_id"])
    update_decisions(winners=winners, threshold=thr, winners_table=table)
    print(f"Ngưỡng Δ = {thr:.4f} (std_T00 = {std:.4f}); yếu tố thắng: {winners or 'không có'}")
    return load_decisions()


def _exp(exp_id: str) -> dict:
    return next(e for e in TRAIN_EXPERIMENTS if e["exp_id"] == exp_id)


def combo_overrides(winners: list[str], table: dict) -> tuple[dict, list[str]]:
    """Ghép yếu tố thắng tốt nhất của mỗi slot, bỏ cái xung đột (giữ cái có Δ lớn hơn)."""
    best_by_slot = {}
    for w in winners:
        slot = table[w]["slot"]
        if slot not in best_by_slot or table[w]["delta"] > table[best_by_slot[slot]]["delta"]:
            best_by_slot[slot] = w
    chosen = list(best_by_slot.values())
    for a, b in CONFLICTS:
        if a in chosen and b in chosen:
            chosen.remove(a if table[a]["delta"] < table[b]["delta"] else b)
    ov = {}
    for w in sorted(chosen):
        ov.update(_exp(w)["overrides"])
    return ov, sorted(chosen)


def run_combo() -> dict | None:
    """T20: kết hợp các yếu tố thắng. Nếu có < 2 yếu tố thắng thì không cần chạy (không có gì để kết hợp)."""
    _need_gpu_warning()
    d = load_decisions()
    winners, table = d["winners"], d["winners_table"]
    ov, chosen = combo_overrides(winners, table)
    update_decisions(combo={"members": chosen, "overrides": ov})
    if len(chosen) < 2:
        print("Có < 2 yếu tố thắng: không chạy T20.", chosen)
        return None
    note = "kết hợp " + " + ".join(f"{c} ({_exp(c)['desc']})" for c in chosen)
    return TR.run(make_cfg("T20", seed=0, desc="combo", note=note, **{**base_overrides(), **ov}))


def decide_recipe() -> dict:
    """Chọn công thức cuối (xem docstring module) và ghi vào decisions.json."""
    d = load_decisions()
    thr = d["threshold"]
    cands = {"T00": ({}, 0, load_summary("T00", 0)["val_macro_f1"])}
    for w in d["winners"]:
        cands[w] = (_exp(w)["overrides"], 1, load_summary(w, 0)["val_macro_f1"])
    s20 = load_summary("T20", 0)
    if s20 and d.get("combo") and len(d["combo"]["members"]) >= 2:
        cands["T20"] = (d["combo"]["overrides"], len(d["combo"]["members"]), s20["val_macro_f1"])
    top = max(v[2] for v in cands.values())
    near = [k for k, v in cands.items() if v[2] >= top - thr]
    pick = min(near, key=lambda k: (cands[k][1], -cands[k][2]))
    # nếu ứng viên vừa chọn không hơn T00 rõ rệt thì quay về T00
    if pick != "T00" and cands[pick][2] - cands["T00"][2] < thr:
        pick = "T00"
    ov = cands[pick][0]
    d = update_decisions(recipe_id=pick, recipe_overrides=ov,
                         recipe_reason=f"ứng viên {{{', '.join(f'{k}: {v[2]:.4f}' for k, v in cands.items())}}}; "
                                       f"chọn {pick} (ngưỡng {thr:.4f})")
    print("Công thức cuối:", pick, ov, "|", d["recipe_reason"])
    return d


# --------------------------------------------------------------------------- #
# Bước 4a: huấn luyện lại cấu hình chung kết (chưa mở test)
# --------------------------------------------------------------------------- #
def run_final_training(seeds=FINAL_SEEDS) -> list[dict]:
    """F01 = công thức cuối trên backbone chính, mỗi seed một lần train. Nếu đã có lần chạy cấu hình GIỐNG HỆT
    (T00 hoặc T-run seed 0 của công thức được chọn) thì dùng lại qua alias_run thay vì train lại."""
    _need_gpu_warning()
    d = load_decisions()
    rid, ov = d["recipe_id"], d["recipe_overrides"]
    out = []
    for sd in seeds:
        have = load_summary(rid, sd)
        if have:
            out.append(TR.alias_run(rid, "F01", sd, PATHS["out_dir"], PATHS["curves_dir"], desc="final",
                                    note=f"cấu hình chung kết (= {rid})"))
        else:
            cfg = make_cfg("F01", seed=sd, desc="final", note=f"cấu hình chung kết (công thức {rid})",
                           **{**base_overrides(), **ov})
            out.append(TR.run(cfg))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Chạy các thí nghiệm của Lab Day 2 (idempotent)")
    for name, default in PATHS.items():
        ap.add_argument("--" + name.replace("_", "-"), default=None, help=f"mặc định {default}")
    ap.add_argument("--epochs", type=int, default=None, help="ghi đè số epoch cho MỌI thí nghiệm (hạ ngân sách)")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--cache-images", action="store_true", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("backbones")
    p.add_argument("--only")
    p = sub.add_parser("select-backbone")
    p.add_argument("--manual")
    sub.add_parser("noise")
    p = sub.add_parser("training")
    p.add_argument("--include-extra", action="store_true")
    p.add_argument("--only")
    sub.add_parser("winners")
    sub.add_parser("combo")
    sub.add_parser("recipe")
    sub.add_parser("final-train")
    return ap


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    set_paths(**{k: getattr(a, k) for k in list(PATHS) + ["num_workers", "cache_images", "epochs"]})
    split = (lambda s: s.split(",") if s else None)
    if a.cmd == "list":
        print("Backbone:", BACKBONE_RUNS)
        for e in TRAIN_EXPERIMENTS:
            print(f"{e['exp_id']} [{e['axis']}] {'core ' if e['core'] else 'extra'} {e['desc']:12s} {e['changes']}")
    elif a.cmd == "backbones":
        run_backbones(split(a.only))
    elif a.cmd == "select-backbone":
        select_backbone(a.manual)
    elif a.cmd == "noise":
        run_noise()
    elif a.cmd == "training":
        run_training(a.include_extra, split(a.only))
    elif a.cmd == "winners":
        decide_winners()
    elif a.cmd == "combo":
        run_combo()
    elif a.cmd == "recipe":
        decide_recipe()
    elif a.cmd == "final-train":
        run_final_training()


if __name__ == "__main__":
    main()
