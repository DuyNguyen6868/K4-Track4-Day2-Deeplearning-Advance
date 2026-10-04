"""eda.py - phân tích dữ liệu và kiểm tra đường ống TRƯỚC khi train (README mục 2.1, RUBRIC mục A/P).

    python eda.py eda   --images-dir data/images --labels-dir data/labels --figures-dir figures
    python eda.py sanity --backbone resnet50 --images-dir ... --labels-dir ... --figures-dir figures

`eda`   : kiểm tra chia dữ liệu (không giao nhau, tổng 17509, tỉ lệ 60/20/20, đủ file), phân phối lớp từng tập,
          đối chiếu với Table 1 của bài báo, >= 3 ảnh mỗi lớp.  Không dùng test cho bất kỳ quyết định nào.
`sanity`: (1) loss ban đầu ~ ln 9 với head khởi tạo mới; (2) overfit một batch nhỏ (loss -> ~0) chứng tỏ mô hình,
          loss, optimizer nối đúng; (3) ảnh sau augmentation để mắt thường kiểm tra nhãn và chuẩn hoá.
Chỉ ĐỌC train/val (và đếm test), không huấn luyện thật, không mở test để chấm.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D  # noqa: E402
import losses as L  # noqa: E402
import model as M  # noqa: E402
import train as TR  # noqa: E402  (cũng đặt stdout UTF-8)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def run_eda(images_dir: str = "data/images", labels_dir: str = "data/labels", figures_dir: str = "figures",
            n_per_class: int = 4, expected_total: int | None = D.EXPECTED_TOTAL, seed: int = 0) -> dict:
    """Kiểm tra chia dữ liệu + thống kê lớp + lưới ảnh. Trả về dict kết quả và ghi vào figures_dir."""
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = D.load_split(labels_dir)
    check = D.check_split(train_df, val_df, test_df, images_dir, expected_total=expected_total)
    D.save_check(check, fig_dir / "split_check.json")

    names = D.CLASS_NAMES
    tbl = pd.DataFrame({"lớp": names,
                        "train": D.class_counts(train_df), "val": D.class_counts(val_df),
                        "test": D.class_counts(test_df)})
    tbl["tổng"] = tbl[["train", "val", "test"]].sum(axis=1)
    tbl["bài báo (Table 1)"] = [D.PAPER_COUNTS[n] for n in names]
    tbl["lệch so với bài báo"] = tbl["tổng"] - tbl["bài báo (Table 1)"]
    tbl["% trong tập train"] = (100 * tbl["train"] / tbl["train"].sum()).round(2)
    tbl["tỉ lệ so với lớp ít nhất"] = (tbl["train"] / tbl["train"].min()).round(2)
    tbl.to_csv(fig_dir / "class_counts.csv", index=False)

    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(names))
    w = 0.27
    for k, (col, c) in enumerate((("train", "C0"), ("val", "C1"), ("test", "C2"))):
        axs[0].bar(x + (k - 1) * w, tbl[col], w, label=col, color=c)
    axs[0].set_yscale("log")
    axs[0].set(xticks=x, title="Số ảnh theo lớp và theo tập (thang log)", ylabel="số ảnh")
    axs[0].set_xticklabels(names, rotation=40, ha="right")
    axs[0].legend()
    axs[1].bar(x, 100 * tbl["train"] / tbl["train"].sum(), color="C0")
    axs[1].set(xticks=x, title="Tỉ lệ lớp trong tập train (%)", ylabel="%")
    axs[1].set_xticklabels(names, rotation=40, ha="right")
    for i, v in enumerate(100 * tbl["train"] / tbl["train"].sum()):
        axs[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "class_distribution.png", dpi=140)
    plt.close(fig)

    # lưới ảnh: n_per_class ảnh ngẫu nhiên (cố định seed) của mỗi lớp, lấy từ train
    from PIL import Image
    rng = np.random.default_rng(seed)
    fig, axs = plt.subplots(len(names), n_per_class, figsize=(2.1 * n_per_class, 2.1 * len(names)), squeeze=False)
    for c, n in enumerate(names):
        pool = train_df.loc[train_df["Label"] == c, "Filename"].tolist()
        pick = rng.choice(pool, size=min(n_per_class, len(pool)), replace=False)
        for j in range(n_per_class):
            a = axs[c][j]
            a.axis("off")
            if j < len(pick):
                with Image.open(Path(images_dir) / pick[j]) as im:
                    a.imshow(im.convert("RGB"))
                if j == 0:
                    a.set_title(n, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "samples_per_class.png", dpi=110)
    plt.close(fig)

    with Image.open(Path(images_dir) / train_df["Filename"].iloc[0]) as im:
        size = im.size
    summary = {"check": check, "class_table": tbl.to_dict("records"), "image_size": list(size),
               "imbalance_ratio_train": float(tbl["train"].max() / tbl["train"].min())}
    (fig_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=int),
                                              encoding="utf-8")
    print(tbl.to_string(index=False))
    print(f"kích thước ảnh: {size}; chênh lệch lớp (lớn nhất/nhỏ nhất, train): {summary['imbalance_ratio_train']:.1f}x")
    return summary


def run_sanity(backbone: str = "resnet50", images_dir: str = "data/images", labels_dir: str = "data/labels",
               figures_dir: str = "figures", img_size: int = 224, batch_size: int = 16, pretrained: bool = True,
               overfit_steps: int = 80, device: str | None = None, seed: int = 0) -> dict:
    """Ba kiểm tra đường ống. Trả về dict (ok bool từng mục) và ghi sanity.json."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    TR.set_seed(seed)
    train_df, _, _ = D.load_split(labels_dir)
    model = M.build_model(backbone, pretrained=pretrained, init="finetune").to(dev)
    cfgm = getattr(model, "pretrained_cfg", {}) or {}
    mean, std = tuple(cfgm.get("mean", D.IMAGENET_MEAN)), tuple(cfgm.get("std", D.IMAGENET_STD))
    crit = L.build_criterion("ce")
    out: dict = {"backbone": M.resolve_name(backbone), "device": str(dev), "mean": mean, "std": std}

    # (1) loss ban đầu ~ ln 9: lấy mẫu cân bằng lớp để tránh lớp Negatives (52%) kéo loss xuống
    tf_eval = D.build_transforms(False, img_size, mean=mean, std=std)
    per = max(2, batch_size // D.NUM_CLASSES + 1)
    sub = pd.concat([g.sample(min(per, len(g)), random_state=seed) for _, g in train_df.groupby("Label")])
    dl = D.make_loader(sub, images_dir, tf_eval, batch_size=len(sub), train=False, num_workers=0)
    xb, yb, _ = next(iter(dl))
    model.eval()
    with torch.no_grad():
        loss0 = float(crit(model(xb.to(dev)).float(), yb.to(dev)))
    out["initial_loss"] = loss0
    out["ln9"] = math.log(D.NUM_CLASSES)
    out["initial_loss_ok"] = abs(loss0 - math.log(D.NUM_CLASSES)) < 1.0  # head mới, logit nhỏ: kỳ vọng ~ 2,2

    # (2) overfit một batch (không augmentation): loss phải về gần 0
    small = sub.iloc[:batch_size]
    dl1 = D.make_loader(small, images_dir, tf_eval, batch_size=len(small), train=False, num_workers=0)
    x1, y1, _ = next(iter(dl1))
    x1, y1 = x1.to(dev), y1.to(dev)
    cfg = TR.Config(lr_backbone=1e-3, lr_head=1e-3, weight_decay=0.0, warmup_epochs=0.0, epochs=1)
    opt = TR.build_optimizer(model, cfg)
    losses = []
    for _ in range(overfit_steps):
        M.set_train_mode(model)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x1).float(), y1)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        acc = float((model(x1).argmax(1) == y1).float().mean())
    out.update({"overfit_first_loss": losses[0], "overfit_last_loss": losses[-1], "overfit_acc_eval": acc,
                "overfit_ok": bool(losses[-1] < 0.1 * losses[0] and losses[-1] < 0.3)})
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(losses)
    ax.set(xlabel="bước", ylabel="loss", title=f"Overfit một batch ({len(small)} ảnh): loss phải về ~0", yscale="log")
    fig.tight_layout()
    fig.savefig(fig_dir / "sanity_overfit.png", dpi=130)
    plt.close(fig)

    # (3) xem trước augmentation (cùng một ảnh, nhiều lần biến đổi, giải chuẩn hoá)
    one = train_df.iloc[[int(train_df.index[train_df["Label"] == 7][0])]]  # một ảnh Snake Weed
    augs = D.AUG_CHOICES
    n_show = 5
    fig, axs = plt.subplots(len(augs), n_show, figsize=(2.0 * n_show, 2.0 * len(augs)), squeeze=False)
    for i, aug in enumerate(augs):
        ds = D.DeepWeedsDataset(one, images_dir, D.build_transforms(True, img_size, aug, mean, std))
        for j in range(n_show):
            img, lab, _ = ds[0]
            axs[i][j].imshow(D.denormalize(img, mean, std).permute(1, 2, 0).numpy())
            axs[i][j].axis("off")
            if j == 0:
                axs[i][j].set_title(f"aug={aug}", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "augmentation_preview.png", dpi=110)
    plt.close(fig)
    out["ok"] = bool(out["initial_loss_ok"] and out["overfit_ok"])
    (fig_dir / "sanity.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eda", "sanity"])
    ap.add_argument("--images-dir", default="data/images")
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    if a.cmd == "eda":
        run_eda(a.images_dir, a.labels_dir, a.figures_dir)
    else:
        run_sanity(a.backbone, a.images_dir, a.labels_dir, a.figures_dir, a.img_size, pretrained=not a.no_pretrained)
