"""study.py - Bước 3: so sánh các phương pháp suy luận (I00-I08) và đo độ trễ, CHỈ trên VAL.

Mô hình khảo sát: F01 seed 0 (công thức cuối). Không huấn luyện lại, không đụng test.
Kết quả: <out_dir>/inference/{inference_table.csv, latency_table.csv, val_views.npz}, hai biểu đồ trong figures/,
và hai quyết định ghi vào decisions.json (đặt TRƯỚC khi mở test):
  - "inference": bộ view + cách gộp cho F01. Chọn bộ có macro-F1 val cao nhất; nếu hơn I00 chưa tới
    MIN_TTA_GAIN thì giữ I00 (không trả K lần chi phí cho phần tăng nằm trong nhiễu). Temperature scaling luôn
    áp dụng (không đổi accuracy, chỉ cải thiện hiệu chuẩn).
  - "realtime": cấu hình nhanh nhất (p95 batch-1) trong số các cấu hình đạt p95 <= ngân sách và có macro-F1 val
    không thấp hơn I00 quá MAX_RT_LOSS (RUBRIC I5).

Chạy:  python study.py [--sizes ...]   (hoặc gọi study.run_study(...) trong notebook)
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B  # noqa: E402
import dataset as D  # noqa: E402
import experiments as X  # noqa: E402
import inference as I  # noqa: E402
import train as TR  # noqa: E402
from evalref import ev  # noqa: E402

MIN_TTA_GAIN = 0.002
MAX_RT_LOSS = 0.002
RT_BUDGET_MS = 100.0


def metrics_from_probs(y: np.ndarray, probs: np.ndarray) -> dict:
    m = ev.compute_metrics(y, probs.argmax(1), probs)
    return {k: m[k] for k in ("macro_f1", "top1", "balanced_acc", "ece", "nll")}


def view_loader(cfg: TR.Config, df: pd.DataFrame, mean, std, batch_size: int = 64):
    """Loader trả ẢNH ĐẦY ĐỦ 256x256 (không cắt); các view tự cắt/resize từ đây."""
    tf = D.build_transforms(False, cfg.img_size, mean=mean, std=std, test_size=256)
    return D.make_loader(df, cfg.images_dir, tf, batch_size, False, None, cfg.num_workers)


def safe_predict_views(model, loader, device, views: dict, dtype: str = "fp32"):
    """predict_views cho tất cả view một lượt; nếu một view lỗi (ví dụ ViT/Swin kích thước cố định) thì chạy từng
    view riêng và trả về thêm danh sách view lỗi."""
    try:
        names, y, out = I.predict_views(model, loader, device, views, dtype=dtype)
        return names, y, out, {}
    except Exception:
        out, failed, names, y = {}, {}, None, None
        for k, fn in views.items():
            try:
                names, y, o = I.predict_views(copy.deepcopy(model), loader, device, {k: fn}, dtype=dtype)
                out.update(o)
            except Exception as e:  # noqa: BLE001
                failed[k] = f"{type(e).__name__}: {str(e)[:80]}"
        return names, y, out, failed


def _ece_crossfit(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """ECE trước/sau TS với T khớp trên nửa này, đo trên nửa kia (2-fold, chẵn/lẻ), lấy trung bình.
    Ước lượng trung thực hơn việc khớp và đo trên cùng val."""
    idx = np.arange(len(y))
    before, after = [], []
    for fit, ev_ in ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)):
        T = I.fit_temperature(logits[fit], y[fit])
        before.append(ev.ece_score(I.softmax_np(logits[ev_]), y[ev_]))
        after.append(ev.ece_score(I.apply_temperature(logits[ev_], T), y[ev_]))
    return float(np.mean(before)), float(np.mean(after))


def plot_reliability(probs_a, probs_b, y, path, title, bins: int = 15) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for a, p, name in zip(ax, (probs_a, probs_b), ("trước temperature scaling", "sau temperature scaling")):
        conf, correct = p.max(1), (p.argmax(1) == y).astype(float)
        idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
        xs, ys, ns = [], [], []
        for m in range(bins):
            mk = idx == m
            if mk.any():
                xs.append(conf[mk].mean())
                ys.append(correct[mk].mean())
                ns.append(mk.sum())
        a.plot([0, 1], [0, 1], "k--", lw=1)
        a.scatter(xs, ys, s=np.array(ns) / max(ns) * 250 + 15, alpha=0.8)
        a.set(xlabel="độ tin cậy (max softmax)", title=f"{name}\nECE = {ev.ece_score(p, y):.4f}")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("độ chính xác trong bin")
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_tradeoff(rows: list[dict], path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = [r for r in rows if r.get("p50_ms") is not None and r.get("macro_f1_val") is not None]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for r in pts:
        ax.scatter(r["p50_ms"], r["macro_f1_val"], s=45)
        ax.annotate(r["exp_id"], (r["p50_ms"], r["macro_f1_val"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set(xlabel="độ trễ p50, batch 1 (ms)", ylabel="macro-F1 val", title=title, xscale="log")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_study(sizes=None, device: str | None = None, lat_warmup: int = 10, lat_iters: int = 100,
              lat_strict: bool = True, skip_latency: bool = False, figures_dir: str = "figures",
              exp: str = "F01") -> dict:
    """Chạy toàn bộ Bước 3 trên val. Trả về dict {"inference_table", "latency_table", "decisions"}."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = X.load_decisions()
    out_dir = Path(X.PATHS["out_dir"]) / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    rd = Path(X.PATHS["out_dir"]) / exp / "seed0"
    model, cfg = TR.load_run_model(rd, dev)
    cfg.images_dir, cfg.labels_dir, cfg.num_workers = X.PATHS["images_dir"], X.PATHS["labels_dir"], X.GLOBAL["num_workers"]
    img = cfg.img_size
    sizes = tuple(sizes) if sizes else (img, 256, 288, 320)
    ms_sizes = sizes[:3]
    mean, std = TR._train_mean_std(model)
    _, val_df, _ = D.load_split(cfg.labels_dir, cfg.fold)
    loader = view_loader(cfg, val_df, mean, std)

    # ---- một lượt đọc val, mọi view ----
    views = I.standard_views(img, sizes)
    names, y, L_, failed = safe_predict_views(model, loader, dev, views)
    np.savez(out_dir / "val_views.npz", files=np.array(names), y=y, **{f"v_{k}": v for k, v in L_.items()})
    center = L_["center"]

    def lat(key, fn):
        if skip_latency:
            return {"p50": None, "p95": None, "p99": None, "images_per_s": None}
        if key not in memo:
            memo[key] = fn()
        return memo[key]

    memo: dict = {}
    lkw = dict(warmup=lat_warmup, iters=lat_iters, strict=lat_strict)

    def single_lat(m, size, dtype="fp32", batch=1, fused=False):
        return lat(("single", id(m), size, dtype, batch, fused),
                   lambda: B.latency_report(m, batch, size, dtype, dev, fused_bn=fused, **lkw))

    base_lat = single_lat(model, img)
    p50_0 = base_lat["p50"]

    rows: list[dict] = []

    def add(exp_id, method, probs, K, lat_r, note="", model_name=None, views_used=None, space=None, extra=None):
        m = metrics_from_probs(y, probs)
        r = {"exp_id": exp_id, "method": method, "model": model_name or f"{exp} seed0", "K": K,
             "macro_f1_val": m["macro_f1"], "top1_val": m["top1"], "ece_val": m["ece"], "nll_val": m["nll"],
             "p50_ms": lat_r["p50"], "p95_ms": lat_r["p95"], "p99_ms": lat_r["p99"],
             "throughput_b1_img_s": lat_r["images_per_s"],
             "cost_vs_I00": (lat_r["p50"] / p50_0) if (lat_r["p50"] is not None and p50_0) else None,
             "note": note, "_views": views_used, "_space": space}
        if extra:
            r.update(extra)
        rows.append(r)
        return m

    def agg(names_, space):
        return I.aggregate_views([L_[n] for n in names_], space)

    def tta_lat(k):
        return lat(("tta", k), lambda: B.tta_latency(model, k, batch_size=1, img_size=img, dtype="fp32",
                                                      device=dev, **lkw))

    def ms_lat():
        """Đa tỉ lệ: tổng độ trễ của các lượt forward ở từng kích thước (tuần tự)."""
        if skip_latency:
            return {"p50": None, "p95": None, "p99": None, "images_per_s": None}
        rs = [single_lat(model, s) for s in ms_sizes]
        return {"p50": sum(r["p50"] for r in rs), "p95": sum(r["p95"] for r in rs),
                "p99": sum(r["p99"] for r in rs), "images_per_s": None}

    # I00, I01, I02, I03
    add("I00", "1 view (mốc): resize 256 + center crop", agg(["center"], "prob"), 1, base_lat,
        views_used=["center"], space="prob")
    sets = {"I01": ("TTA lật ngang (K=2)", ["center", "center_flip"], 2),
            "I02a": ("TTA 5 crop (K=5)", ["center", "tl", "tr", "bl", "br"], 5),
            "I02b": ("TTA 5 crop + lật (K=10)", ["center", "tl", "tr", "bl", "br", "center_flip", "tl_flip",
                                                   "tr_flip", "bl_flip", "br_flip"], 10),
            "I02c": (f"TTA đa tỉ lệ {ms_sizes} (K=3)", [f"full{s}" for s in ms_sizes], 3)}
    for eid, (label, vs, k) in sets.items():
        miss = [v for v in vs if v not in L_]
        if miss:
            rows.append({"exp_id": eid, "method": label, "model": f"{exp} seed0", "K": k,
                         "note": f"không áp dụng: {', '.join(failed.get(v, v) for v in miss)}"})
            continue
        lr_ = ms_lat() if eid == "I02c" else tta_lat(k)
        add(eid, label + " [gộp xác suất]", agg(vs, "prob"), k, lr_, views_used=vs, space="prob")
    for eid, base in (("I03a", "I01"), ("I03b", "I02a"), ("I03c", "I02b"), ("I03d", "I02c")):
        label, vs, k = sets[base]
        if all(v in L_ for v in vs):
            src = next(r for r in rows if r["exp_id"] == base)
            add(eid, label + " [gộp logit]", agg(vs, "logit"), k,
                {"p50": src["p50_ms"], "p95": src["p95_ms"], "p99": src["p99_ms"], "images_per_s": src["throughput_b1_img_s"]},
                note=f"so với {base}: cùng bộ view, khác cách gộp", views_used=vs, space="logit")
    # I04: độ phân giải kiểm tra (toàn ảnh resize)
    for letter, s in zip("abcd", sizes):
        v = f"full{s}"
        if v in L_:
            add(f"I04{letter}", f"độ phân giải kiểm tra {s} (toàn ảnh)", agg([v], "prob"), 1,
                single_lat(model, s), note="không đổi tham số; FLOPs tăng theo s^2", views_used=[v], space="prob")
        else:
            rows.append({"exp_id": f"I04{letter}", "method": f"độ phân giải kiểm tra {s}", "model": f"{exp} seed0",
                         "K": 1, "note": f"không áp dụng: {failed.get(v, 'lỗi')}"})
    # I05: ensemble
    seeds = [s for s in X.FINAL_SEEDS if (Path(X.PATHS["out_dir"]) / exp / f"seed{s}" / "val_logits.npz").exists()]
    if len(seeds) >= 2:
        probs_list, lat_sum = [], 0.0
        for s in seeds:
            z = np.load(Path(X.PATHS["out_dir"]) / exp / f"seed{s}" / "val_logits.npz")
            assert z["files"].tolist() == names, "thứ tự file val khác nhau giữa các seed"
            probs_list.append(I.softmax_np(z["logits"]))
            lat_sum += base_lat["p50"] or 0.0
        add("I05a", f"ensemble {len(seeds)} seed của {exp} (trung bình xác suất)", I.ensemble_probs(probs_list),
            len(seeds), {"p50": lat_sum or None, "p95": None, "p99": None, "images_per_s": None},
            model_name=f"{exp} seeds {seeds}", note="chi phí = số mô hình")
    light = d.get("backbone_light")
    if light:
        eid_l = next(e for e, k in X.BACKBONE_RUNS.items() if k == light)
        zl = Path(X.PATHS["out_dir"]) / eid_l / "seed0" / "val_logits.npz"
        if zl.exists() and light != cfg.backbone:
            z = np.load(zl)
            assert z["files"].tolist() == names
            lm, _ = TR.load_run_model(Path(X.PATHS["out_dir"]) / eid_l / "seed0", dev)
            ll = single_lat(lm, img)
            add("I05b", f"ensemble {exp} + {light} ({eid_l}, công thức nền)",
                I.ensemble_probs([I.softmax_np(center), I.softmax_np(z["logits"])]), 2,
                {"p50": (base_lat["p50"] + ll["p50"]) if base_lat["p50"] is not None else None, "p95": None,
                 "p99": None, "images_per_s": None}, model_name=f"{exp} seed0 + {eid_l}",
                note="khác backbone; chi phí = tổng hai mô hình")
    # I06: EMA
    for src_id in (exp, "T12"):
        s = X.load_summary(src_id, 0)
        if s and s.get("val_macro_f1_raw") is not None:
            rows.append({"exp_id": "I06", "method": "trọng số EMA (so với trọng số thô cùng lần chạy)",
                         "model": f"{src_id} seed0", "K": 1, "macro_f1_val": s["val_macro_f1"],
                         "top1_val": s["val_top1"], "ece_val": s["val_ece"],
                         "p50_ms": base_lat["p50"], "p95_ms": base_lat["p95"], "p99_ms": base_lat["p99"],
                         "cost_vs_I00": 1.0,
                         "note": f"macro-F1 val trọng số thô = {s['val_macro_f1_raw']:.4f}; không tốn thêm khi suy luận"})
            break
    # I07: temperature scaling
    T = I.fit_temperature(center, y)
    xb, xa = _ece_crossfit(center, y)
    add("I07", f"temperature scaling (T = {T:.3f}, khớp trên val)", I.apply_temperature(center, T), 1, base_lat,
        note=f"ECE val 2-fold chéo: {xb:.4f} -> {xa:.4f} (ước lượng không khớp-và-đo trên cùng tập); "
             f"accuracy không đổi", extra={"temperature": T, "ece_crossfit_before": xb, "ece_crossfit_after": xa})
    plot_reliability(I.softmax_np(center), I.apply_temperature(center, T), y,
                     Path(figures_dir) / "reliability_val.png", f"{exp} seed0, val (T = {T:.3f})")

    # I08: gộp BN, FP16, AMP (độ chính xác + độ trễ)
    fused = I.fuse_conv_bn(model, check_size=min(img, 64))
    can_fuse = fused.fuse_info["n_fused"] > 0
    center_loader = view_loader(cfg, val_df, mean, std)
    cv = {"center": views["center"]}

    def variant(m, dtype):
        _, _, o = I.predict_views(copy.deepcopy(m), center_loader, dev, cv, dtype=dtype)
        return I.softmax_np(o["center"])

    variants = [("I08a", "gộp BN vào conv (FP32)", fused, "fp32", True),
                ("I08b", "FP16 (model.half())", model, "fp16", False),
                ("I08c", "AMP (autocast)", model, "amp", False),
                ("I08d", "FP16 + gộp BN", fused, "fp16", True)]
    for eid, label, m, dtype, fz in variants:
        if fz and not can_fuse:
            rows.append({"exp_id": eid, "method": label, "model": f"{exp} seed0", "K": 1,
                         "note": "không áp dụng: kiến trúc không có BatchNorm (LayerNorm)"})
            continue
        add(eid, label, variant(m, dtype), 1, single_lat(m, img, dtype, 1, fz),
            note=f"max|Δlogit| sau gộp BN = {fused.fuse_info['max_abs_err']:.1e}" if fz else "",
            extra={"dtype": dtype, "fused_bn": fz})

    # ---- bảng độ trễ ----
    lat_rows = []
    if not skip_latency:
        for dtype in ("fp32", "amp", "fp16"):
            for fz in ([False, True] if can_fuse else [False]):
                for bs in (1, 32):
                    r = single_lat(fused if fz else model, img, dtype, bs, fz)
                    lat_rows.append({"config": f"{exp} {dtype}{' + gộp BN' if fz else ''}", **r})
        for k in (2, 5, 10):
            r = tta_lat(k)
            lat_rows.append({"config": f"{exp} TTA K={k} (tuần tự, fp32)", **r})
        for s in sizes:
            lat_rows.append({"config": f"{exp} độ phân giải {s} (fp32)", **single_lat(model, s)})
        if light and light != cfg.backbone:
            eid_l = next(e for e, k in X.BACKBONE_RUNS.items() if k == light)
            lm, _ = TR.load_run_model(Path(X.PATHS["out_dir"]) / eid_l / "seed0", dev)
            for dtype in ("fp32", "fp16"):
                lat_rows.append({"config": f"{eid_l} {light} {dtype}", **single_lat(lm, img, dtype)})
    lat_df = pd.DataFrame(lat_rows)
    if len(lat_df):
        lat_df.to_csv(out_dir / "latency_table.csv", index=False)

    # ---- quyết định (chỉ dựa trên val) ----
    cand = [r for r in rows if r.get("_views") and r["exp_id"] in
            ("I00", "I01", "I02a", "I02b", "I02c", "I03a", "I03b", "I03c", "I03d")]
    i00 = next(r for r in cand if r["exp_id"] == "I00")
    best = max(cand, key=lambda r: (r["macro_f1_val"], -(r["K"] or 1)))
    pick = best if best["macro_f1_val"] - i00["macro_f1_val"] >= MIN_TTA_GAIN else i00
    decision_inf = {"set_id": pick["exp_id"], "views": pick["_views"], "space": pick["_space"],
                    "gain_vs_I00": pick["macro_f1_val"] - i00["macro_f1_val"],
                    "reason": (f"{pick['exp_id']} có macro-F1 val cao nhất ({pick['macro_f1_val']:.4f}), hơn I00 "
                               f"{pick['macro_f1_val'] - i00['macro_f1_val']:+.4f} >= {MIN_TTA_GAIN}"
                               if pick is not i00 else
                               f"không bộ TTA nào hơn I00 >= {MIN_TTA_GAIN} (tốt nhất {best['exp_id']} "
                               f"{best['macro_f1_val'] - i00['macro_f1_val']:+.4f}); giữ 1 view")}
    decision_rt = None
    if not skip_latency:
        opts = []
        for eid in ("I00", "I08a", "I08b", "I08c", "I08d"):
            r = next((x for x in rows if x["exp_id"] == eid and x.get("macro_f1_val") is not None), None)
            if r is None or r["p95_ms"] is None:
                continue
            if r["macro_f1_val"] >= i00["macro_f1_val"] - MAX_RT_LOSS:
                opts.append(r)
        ok = [r for r in opts if r["p95_ms"] <= RT_BUDGET_MS]
        chosen = min(ok or opts, key=lambda r: r["p95_ms"])
        decision_rt = {"exp_id": chosen["exp_id"], "method": chosen["method"], "dtype": chosen.get("dtype", "fp32"),
                       "fused_bn": bool(chosen.get("fused_bn", False)), "p50_ms": chosen["p50_ms"],
                       "p95_ms": chosen["p95_ms"], "p99_ms": chosen["p99_ms"],
                       "val_macro_f1": chosen["macro_f1_val"], "within_budget": bool(ok), "gpu": base_lat.get("gpu"),
                       "latency_includes_preprocessing": False}
    X.update_decisions(inference=decision_inf, realtime=decision_rt, temperature_note="T khớp riêng cho từng seed trên val")

    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    table.to_csv(out_dir / "inference_table.csv", index=False)
    plot_tradeoff(rows, Path(figures_dir) / "tradeoff_f1_vs_latency.png",
                  f"Đánh đổi macro-F1 val và độ trễ batch-1 ({exp} seed0)")
    print(table[["exp_id", "method", "K", "macro_f1_val", "top1_val", "ece_val", "p50_ms", "cost_vs_I00"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))
    print("Quyết định suy luận:", json.dumps(decision_inf, ensure_ascii=False))
    print("Cấu hình thời gian thực:", json.dumps(decision_rt, ensure_ascii=False))
    return {"inference_table": table, "latency_table": lat_df, "decisions": X.load_decisions()}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    a = ap.parse_args()
    X.set_paths(out_dir=a.out_dir, images_dir=a.images_dir, labels_dir=a.labels_dir)
    run_study(a.sizes)
