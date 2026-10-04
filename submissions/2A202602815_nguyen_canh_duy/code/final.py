"""final.py - Bước 4: chạy TEST đúng một lần mỗi seed cho cấu hình chung kết (F01) và mốc (T00 + I00).

Điều kiện: đã có F01 và T00 cho các seed (experiments.run_final_training, run_noise) và đã chốt quyết định suy
luận trên val (study.run_study). Mọi quyết định đọc từ decisions.json; KHÔNG chọn lại gì ở bước này.

Với mỗi seed k:
  F01 : khớp T trên VAL của chính seed đó -> một lượt qua test với mọi view cần dùng -> ghi
        predictions/F01_seed<k>_test.csv        (có temperature scaling)
        predictions/F01_uncal_seed<k>_test.csv  (chưa TS, để chấm I4a)
        predictions/F01_seed<k>_val.csv         (val, để chấm I4b)
        predictions/F01rt_seed<k>_test.csv      (1 view center FP32 của cùng mô hình: cấu hình thời gian thực)
  T00 : 1 view center, không TS -> predictions/T00_seed<k>_test.csv (mốc)
Chốt chặn: train.begin_test ghi test_started.json trong run_dir; lần thứ hai bị từ chối.

Sau đó `score_and_grade` gọi eval.py gốc (score + grade) và `error_analysis` vẽ ma trận nhầm lẫn, ảnh bị đoán sai.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D  # noqa: E402
import experiments as X  # noqa: E402
import inference as I  # noqa: E402
import study as S  # noqa: E402
import train as TR  # noqa: E402
from evalref import EVAL_PY_PATH, ev  # noqa: E402


def views_for(names: list[str], img: int) -> dict:
    sizes = sorted({int(n[4:]) for n in names if n.startswith("full")})
    allv = I.standard_views(img, sizes or (img,))
    missing = [n for n in names if n not in allv]
    if missing:
        raise KeyError(f"view không tồn tại: {missing}")
    return {n: allv[n] for n in names}


def _model_for(exp: str, seed: int, dev: str):
    rd = Path(X.PATHS["out_dir"]) / exp / f"seed{seed}"
    if not (rd / "best.pt").exists():
        raise FileNotFoundError(f"{rd}/best.pt chưa có: huấn luyện {exp} seed {seed} trước")
    model, cfg = TR.load_run_model(rd, dev)
    cfg.images_dir, cfg.labels_dir, cfg.num_workers = X.PATHS["images_dir"], X.PATHS["labels_dir"], X.GLOBAL["num_workers"]
    return rd, model, cfg


def run_test_once(seeds=X.FINAL_SEEDS, device: str | None = None, final_exp: str = "F01",
                  base_exp: str = "T00") -> dict:
    """Mở test: đúng một lần cho mỗi (exp, seed). Trả về dict nhiệt độ T theo seed."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = X.load_decisions()
    if "inference" not in d:
        raise RuntimeError("chưa có quyết định suy luận: chạy study.run_study (trên val) trước")
    inf = d["inference"]
    pred_dir = Path(X.PATHS["pred_dir"])
    temps = {}
    for sd in seeds:
        # ---- F01 ----
        rd, model, cfg = _model_for(final_exp, sd, dev)
        mean, std = TR._train_mean_std(model)
        _, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
        need = sorted(set(inf["views"]) | {"center"})
        views = views_for(need, cfg.img_size)
        vnames, vy, VL = I.predict_views(model, S.view_loader(cfg, val_df, mean, std), dev, views)
        T = I.fit_temperature_views([VL[v] for v in inf["views"]], vy, inf["space"])
        temps[sd] = T
        val_probs = I.aggregate_views([VL[v] / T for v in inf["views"]], inf["space"])
        ev.save_predictions(pred_dir / f"{final_exp}_seed{sd}_val.csv", vnames, vy, val_probs)

        TR.begin_test(rd, who="final.run_test_once")  # từ đây test đã bị mở cho (F01, seed)
        tnames, ty, TL = I.predict_views(model, S.view_loader(cfg, test_df, mean, std), dev, views)
        uncal = I.aggregate_views([TL[v] for v in inf["views"]], inf["space"])
        cal = I.aggregate_views([TL[v] / T for v in inf["views"]], inf["space"])
        ev.save_predictions(pred_dir / f"{final_exp}_seed{sd}_test.csv", tnames, ty, cal)
        ev.save_predictions(pred_dir / f"{final_exp}_uncal_seed{sd}_test.csv", tnames, ty, uncal)
        ev.save_predictions(pred_dir / f"{final_exp}rt_seed{sd}_test.csv", tnames, ty, I.softmax_np(TL["center"]))
        np.savez(rd / "test_views_logits.npz", files=np.array(tnames), y=ty, **{f"v_{k}": v for k, v in TL.items()})
        TR.finish_test(rd, {"n": int(len(ty)), "views": inf["views"], "space": inf["space"], "T": T,
                            "via": "final.run_test_once"})
        # ---- mốc T00 + I00 ----
        rd0, model0, cfg0 = _model_for(base_exp, sd, dev)
        mean0, std0 = TR._train_mean_std(model0)
        TR.begin_test(rd0, who="final.run_test_once")
        n0, y0, L0 = I.predict_views(model0, S.view_loader(cfg0, test_df, mean0, std0), dev,
                                     views_for(["center"], cfg0.img_size))
        ev.save_predictions(pred_dir / f"{base_exp}_seed{sd}_test.csv", n0, y0, I.softmax_np(L0["center"]))
        np.savez(rd0 / "test_views_logits.npz", files=np.array(n0), y=y0, v_center=L0["center"])
        TR.finish_test(rd0, {"n": int(len(y0)), "views": ["center"], "via": "final.run_test_once"})
        print(f"seed {sd}: xong F01 (T = {T:.3f}) và {base_exp}")
    X.update_decisions(temperatures=temps, test_opened=True)
    return temps


@contextlib.contextmanager
def _utf8_write_text():
    """eval.py ghi JSON bằng Path.write_text không đặt encoding: trên Windows (cp1252) nó nổ UnicodeEncodeError
    (một ValueError, nên main() trả mã 2). Không được sửa eval.py, nên mặc định UTF-8 trong lúc gọi nó."""
    orig = Path.write_text

    def patched(self, data, encoding=None, errors=None, newline=None):
        return orig(self, data, encoding=encoding or "utf-8", errors=errors, newline=newline)

    Path.write_text = patched
    try:
        yield
    finally:
        Path.write_text = orig


def _capture(argv: list[str]) -> str:
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err), _utf8_write_text():
        code = ev.main(argv)
    out = buf.getvalue()
    if code != 0:
        raise RuntimeError(f"eval.py {' '.join(argv[:2])} thoát với mã {code}:\n{out}\n{err.getvalue()}")
    return out


def score_and_grade(out_dir: str = "eval_out", final_exp: str = "F01", base_exp: str = "T00") -> dict:
    """Gọi eval.py gốc: `score` cho F01, T00, F01_uncal, F01rt và `grade` (phần I). Lưu markdown vào out_dir."""
    lab, pred = Path(X.PATHS["labels_dir"]), Path(X.PATHS["pred_dir"])
    test_csv, val_csv, labels = lab / "test_subset0.csv", lab / "val_subset0.csv", lab / "labels.csv"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    texts = {}
    for tag, pat in ((final_exp, f"{final_exp}_seed*_test.csv"), (base_exp, f"{base_exp}_seed*_test.csv"),
                     (f"{final_exp}uncal", f"{final_exp}_uncal_seed*_test.csv"),
                     (f"{final_exp}rt", f"{final_exp}rt_seed*_test.csv")):
        texts[tag] = _capture(["score", "--pred", str(pred / pat), "--test-csv", str(test_csv),
                               "--labels", str(labels), "--tag", tag, "--out", out_dir])
        (Path(out_dir) / f"score_{tag}.md").write_text(texts[tag], encoding="utf-8")
    rt = X.load_decisions().get("realtime")
    gargs = ["grade", "--final", str(pred / f"{final_exp}_seed*_test.csv"),
             "--baseline", str(pred / f"{base_exp}_seed*_test.csv"),
             "--uncal", str(pred / f"{final_exp}_uncal_seed*_test.csv"),
             "--final-val", str(pred / f"{final_exp}_seed*_val.csv"),
             "--test-csv", str(test_csv), "--val-csv", str(val_csv), "--labels", str(labels), "--out", out_dir]
    if rt and rt.get("p95_ms") is not None:
        gargs += ["--latency-p95-ms", f"{rt['p95_ms']:.3f}", "--latency-method", "proper"]
    texts["grade"] = _capture(gargs)
    (Path(out_dir) / "grade_I.md").write_text(texts["grade"], encoding="utf-8")
    for key in (final_exp, base_exp, "grade"):
        print(texts[key])
    return texts


def error_analysis(final_exp: str = "F01", figures_dir: str = "figures", out_dir: str = "eval_out",
                   seed: int = 0, max_images: int = 8) -> dict:
    """Ma trận nhầm lẫn (tổng các seed), cặp nhầm nhiều nhất, và ảnh Chinee Apple <-> Snake Weed bị đoán sai.

    Chỉ để PHÂN TÍCH sau khi đã có test; không được dùng để sửa cấu hình rồi chạy lại test.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = Path(X.PATHS["labels_dir"]) / "labels.csv"
    names = ev.load_names(str(labels)) if labels.exists() else list(ev.CLASS_NAMES)
    g = ev.load_group(str(Path(X.PATHS["pred_dir"]) / f"{final_exp}_seed*_test.csv"), None)
    cm = sum(m["confusion"] for m in g.metrics)
    pct = cm / cm.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.1f}%", ha="center", va="center", fontsize=6.5,
                    color="white" if pct[i, j] > 55 else "black")
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xlabel="dự đoán", ylabel="nhãn thật",
           title=f"Ma trận nhầm lẫn test, {final_exp}, tổng {len(g.preds)} seed (số ảnh, % theo hàng)")
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    fig.colorbar(im, fraction=0.04)
    fig.tight_layout()
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(figures_dir) / "confusion_test.png", dpi=140)
    plt.close(fig)

    pairs = [(int(cm[i, j]), names[i], names[j], float(pct[i, j]))
             for i in range(len(names)) for j in range(len(names)) if i != j and cm[i, j] > 0]
    top = pd.DataFrame(sorted(pairs, reverse=True)[:10], columns=["số ảnh (tổng seed)", "nhãn thật", "dự đoán", "% hàng"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    top.to_csv(Path(out_dir) / "top_confusions.csv", index=False)

    # ảnh bị đoán sai của seed đã chọn, ưu tiên cặp Chinee Apple <-> Snake Weed
    p = ev.read_pred(str(Path(X.PATHS["pred_dir"]) / f"{final_exp}_seed{seed}_test.csv"))
    hard = ev.hard_class_indices(names)  # so khớp tên không phân biệt hoa thường
    ca, sw = hard["Chinee Apple"], hard["Snake Weed"]
    wrong = p.y_pred != p.y_true
    sel = [(f, int(t), int(q), float(pr[q])) for f, t, q, pr, w in zip(p.filenames, p.y_true, p.y_pred, p.probs, wrong)
           if w and {int(t), int(q)} == {ca, sw}][:max_images]
    if len(sel) < max_images:  # bổ sung bằng các lỗi tự tin nhất của lớp khác
        rest = sorted([(f, int(t), int(q), float(pr[q])) for f, t, q, pr, w in zip(p.filenames, p.y_true, p.y_pred, p.probs, wrong)
                       if w and (f, int(t), int(q), float(pr[q])) not in sel], key=lambda r: -r[3])
        sel += rest[:max_images - len(sel)]
    if sel:
        from PIL import Image
        cols = 4
        rows_n = int(np.ceil(len(sel) / cols))
        fig, axs = plt.subplots(rows_n, cols, figsize=(3.2 * cols, 3.4 * rows_n), squeeze=False)
        for a in axs.ravel():
            a.axis("off")
        for a, (f, t, q, pr) in zip(axs.ravel(), sel):
            a.imshow(Image.open(Path(X.PATHS["images_dir"]) / f).convert("RGB"))
            a.set_title(f"thật: {names[t]}\nđoán: {names[q]} ({pr:.2f})", fontsize=8)
        fig.suptitle(f"Ảnh test bị đoán sai ({final_exp}, seed {seed}); ưu tiên cặp Chinee Apple <-> Snake Weed")
        fig.tight_layout()
        fig.savefig(Path(figures_dir) / "errors_test.png", dpi=130)
        plt.close(fig)
        pd.DataFrame(sel, columns=["Filename", "y_true", "y_pred", "p_pred"]).to_csv(
            Path(out_dir) / f"errors_seed{seed}.csv", index=False)
    print(top.to_string(index=False))
    return {"confusion_sum": cm, "top_confusions": top, "n_error_images": len(sel)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Bước 4: chạy test một lần mỗi seed rồi chấm bằng eval.py")
    ap.add_argument("--out-dir")
    ap.add_argument("--images-dir")
    ap.add_argument("--labels-dir")
    ap.add_argument("--pred-dir")
    ap.add_argument("--yes-open-test", action="store_true", help="xác nhận đã chốt cấu hình trên val")
    a = ap.parse_args()
    X.set_paths(out_dir=a.out_dir, images_dir=a.images_dir, labels_dir=a.labels_dir, pred_dir=a.pred_dir)
    if not a.yes_open_test:
        sys.exit("Thêm --yes-open-test để mở test (mỗi seed chỉ một lần). Đã chốt cấu hình trên val chưa?")
    run_test_once()
    score_and_grade()
    error_analysis()
    print("eval.py dùng:", EVAL_PY_PATH)
