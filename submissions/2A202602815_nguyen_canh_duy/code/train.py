"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Một hàm `run(cfg)` dùng chung cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0

Chỉ số chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc (qua evalref), để cùng
định nghĩa với lúc chấm. Mặc định run() KHÔNG đụng vào test (quy tắc S4); test chỉ được mở ở Bước 4,
đúng một lần mỗi seed, và có chốt chặn ghi lại việc đó (xem `begin_test`).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

for _s in (sys.stdout, sys.stderr):  # log tiếng Việt không được làm sập chương trình trên console cp1252 (Windows)
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B  # noqa: E402
import dataset as D  # noqa: E402
import losses as L  # noqa: E402
import model as M  # noqa: E402
from evalref import ev  # noqa: E402
from inference import softmax_np  # noqa: E402


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    desc: str = ""                    # mô tả ngắn, dùng đặt tên ảnh curves/<exp_id>_<desc>.png
    note: str = ""                    # ghi chú tự do (khác nền ở điểm nào...)
    # --- mô hình ---
    backbone: str = "resnet50"        # khoá trong model.SUGGESTED_BACKBONES hoặc tên timm
    init: str = "finetune"            # scratch | frozen | finetune
    pretrained: bool = True           # False chỉ để chạy thử offline
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0       # stochastic depth (trục F)
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | mildcrop | geo | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    mix_prob: float = 1.0             # xác suất trộn mỗi batch
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    optimizer: str = "adamw"          # adamw | sgd
    momentum: float = 0.9
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    grad_clip: float | None = None
    ema_decay: float | None = None
    amp: bool = True
    channels_last: bool = True
    num_workers: int = 2
    cache_images: bool = False
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"
    # --- vận hành ---
    save_ckpt: bool = True
    resume: bool = True               # tiếp tục từ last.pt nếu phiên bị ngắt
    overwrite: bool = False           # False: bỏ qua nếu run_dir đã có summary.json
    measure_latency: bool = True      # đo độ trễ batch-1 sơ bộ ở cuối (Bước 1)
    limit_batches: int | None = None  # chỉ để chạy thử nhanh
    expected_total: int | None = D.EXPECTED_TOTAL
    save_val_predictions: bool = False
    # --- chỉ bật nếu muốn ghi predictions TEST ngay trong run() (xem final.py cho đường chính thức) ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    name = f"{cfg.exp_id}_{cfg.desc or cfg.backbone}" + (f"_seed{cfg.seed}" if cfg.seed else "")
    return Path(cfg.curves_dir) / f"{name}.png"


def set_seed(seed: int) -> None:
    """Cố định random, numpy, torch (CPU và CUDA). cudnn.benchmark=False; KHÔNG ép deterministic hoàn toàn
    (AMP và vài toán tử CUDA không tất định), nên chạy lại cho kết quả gần nhưng không nhất thiết trùng bit.
    Seed của worker DataLoader đặt trong dataset.make_loader."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW (hoặc SGD momentum) với 3 nhóm tham số (xem model.param_groups)."""
    groups = M.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=cfg.momentum, nesterov=True)
    raise ValueError(f"optimizer={cfg.optimizer!r} không hợp lệ (adamw | sgd)")


def lr_factor(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 1e-3) -> float:
    """Hệ số nhân LR tại bước `step`: warmup tuyến tính rồi cosine về min_ratio (slide trang 55)."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine, cập nhật THEO BƯỚC (iteration), gọi scheduler.step() sau mỗi optimizer.step()."""
    total = cfg.epochs * steps_per_epoch
    warm = int(round(cfg.warmup_epochs * steps_per_epoch))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total, warm))


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    Giữ một bản sao `self.module` (dùng để đánh giá). Buffer của BatchNorm (running_mean/var) cũng được lấy
    trung bình như tham số; num_batches_tracked được chép nguyên. d được hâm nóng: d_eff = min(d, (1+n)/(10+n))
    để EMA không bị kéo lại bởi trọng số khởi tạo ở những bước đầu.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.n = 0

    @torch.no_grad()
    def update(self, model) -> None:
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        src = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(src[k])


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None, epoch: int = 0) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "train_acc", "lr"}.

    train_acc chỉ tính trên các batch KHÔNG bị trộn (Mixup/CutMix làm accuracy lúc train mất nghĩa).
    """
    M.set_train_mode(model)
    use_amp = cfg.amp and device.type == "cuda"
    rng = np.random.default_rng([cfg.seed, epoch])
    loss_sum, n_seen, correct, n_clean = 0.0, 0, 0, 0
    for step, (x, y, _) in enumerate(loader):
        if cfg.limit_batches is not None and step >= cfg.limit_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg.channels_last and device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        mixed = bool(cfg.mix) and rng.random() < cfg.mix_prob
        if mixed:
            x, targets = L.mix_batch(x, y, cfg.mix_alpha, cfg.mix, rng=rng)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        loss = L.mixed_loss(criterion, logits, targets) if mixed else criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss không hữu hạn ở epoch {epoch}, bước {step}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        bs = y.size(0)
        loss_sum += float(loss.item()) * bs
        n_seen += bs
        if not mixed:
            correct += int((logits.argmax(1) == y).sum())
            n_clean += bs
    return {"train_loss": loss_sum / max(n_seen, 1),
            "train_acc": correct / n_clean if n_clean else float("nan"),
            "lr": optimizer.param_groups[0]["lr"]}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, limit_batches: int | None = None):
    """Chạy model trên một loader ở chế độ eval, KHÔNG gradient, FP32.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader để ghép logit với tên file.
    """
    model.eval()
    names, ys, zs, loss_sum, n = [], [], [], 0.0, 0
    for step, (x, y, f) in enumerate(loader):
        if limit_batches is not None and step >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        z = model(x).float()
        yt = y.to(device)
        loss_sum += float(criterion(z, yt).item()) * len(y)
        n += len(y)
        zs.append(z.cpu().numpy())
        ys.append(np.asarray(y))
        names.extend(f)
    return names, np.concatenate(ys), np.concatenate(zs), loss_sum / max(n, 1)


def plot_curves(history: list[dict], path: str | Path, title: str, lr_steps=None) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    Ba ô: loss (train theo loss huấn luyện, val theo CE), macro-F1 và top-1 val, LR theo bước (nếu có)
    hoặc theo epoch. Gạch đứng đánh dấu epoch tốt nhất.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    ep = h["epoch"] + 1
    best = int(h["val_macro_f1"].values.argmax()) + 1
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(ep, h["train_loss"], "o-", label="train (loss huấn luyện)")
    ax[0].plot(ep, h["val_loss"], "s-", label="val (CE)")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[1].plot(ep, h["val_macro_f1"], "o-", color="C2", label="val macro-F1")
    ax[1].plot(ep, h["val_top1"], "s--", color="C3", label="val top-1")
    if h["train_acc"].notna().any():
        ax[1].plot(ep, h["train_acc"], "^:", color="C4", label="train acc (batch không trộn)")
    ax[1].set(xlabel="epoch", ylabel="chỉ số", title="Chỉ số val")
    if lr_steps is not None:
        ax[2].plot(np.arange(len(lr_steps)), lr_steps)
        ax[2].set(xlabel="bước (iteration)", ylabel="hệ số LR", title="LR: warmup + cosine")
    else:
        ax[2].plot(ep, h["lr"], "o-")
        ax[2].set(xlabel="epoch", ylabel="LR (nhóm 0)", title="LR")
    for a in ax[:2]:
        a.axvline(best, color="gray", ls="--", lw=1, label=f"tốt nhất: epoch {best}")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Chốt chặn test (S4): mỗi (exp_id, seed) chỉ được mở test đúng một lần
# --------------------------------------------------------------------------- #
def begin_test(rd: Path, who: str = "") -> None:
    """Ghi `test_started.json`; ném lỗi nếu test của run này đã từng được mở."""
    started, done = rd / "test_started.json", rd / "test_done.json"
    if started.exists() or done.exists():
        raise RuntimeError(
            f"Test của {rd} ĐÃ được mở trước đó (có {started.name}/{done.name}). "
            "Quy tắc: test chỉ chạy một lần mỗi seed; không chạy lại để chọn kết quả tốt hơn. "
            "Nếu lần trước hỏng vì lỗi kỹ thuật, xoá file đó thủ công và ghi lý do vào báo cáo.")
    started.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "who": who}), encoding="utf-8")


def finish_test(rd: Path, info: dict) -> None:
    (rd / "test_done.json").write_text(json.dumps(
        {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **info}, indent=2), encoding="utf-8")


def load_run_model(rd: str | Path, device="cpu", which: str = "best"):
    """Dựng lại model của một lần chạy từ config.json + best.pt (không tải lại trọng số tiền huấn luyện).

    Trả về (model ở eval, Config). Với run có EMA, best.pt chứa trọng số EMA (đúng thứ được chọn theo val).
    """
    rd = Path(rd)
    raw = json.loads((rd / "config.json").read_text(encoding="utf-8"))
    cfg = Config(**{k: v for k, v in raw.items() if k in Config.__dataclass_fields__})
    model = M.build_model(M.resolve_name(cfg.backbone), pretrained=False, drop_rate=cfg.drop_rate,
                          init="finetune", drop_path_rate=0.0)
    ckpt = torch.load(rd / f"{which}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.weight_tag = ckpt.get("weight_tag", "")
    return model.to(device).eval(), cfg


def _train_mean_std(model):
    c = getattr(model, "pretrained_cfg", {}) or {}
    return tuple(c.get("mean", D.IMAGENET_MEAN)), tuple(c.get("std", D.IMAGENET_STD))


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    Lưu trong run_dir(cfg): config.json, history.csv, best.pt (trọng số tốt nhất theo MACRO-F1 VAL, hòa thì lấy
    epoch sớm hơn; nếu có EMA thì là trọng số EMA), last.pt (để resume), val_logits.npz, summary.json,
    và ảnh curves/<exp_id>_<desc>.png. KHÔNG đụng test trừ khi cfg.save_test_predictions.
    """
    rd = run_dir(cfg)
    if (rd / "summary.json").exists() and not cfg.overwrite:
        print(f"[{cfg.exp_id} seed{cfg.seed}] đã có summary.json, bỏ qua (overwrite=True để chạy lại)")
        return json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    if cfg.mix not in (None, "mixup", "cutmix"):
        raise ValueError(f"mix={cfg.mix!r} không hợp lệ")
    rd.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    (rd / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"

    # 2) dữ liệu và kiểm tra chia
    train_df, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    chk_path = Path(cfg.out_dir) / "split_check.json"
    first = not chk_path.exists()
    chk = D.check_split(train_df, val_df, test_df, cfg.images_dir, expected_total=cfg.expected_total,
                        check_files=first, verbose=first)
    if first:
        D.save_check(chk, chk_path)

    # 3-4) model, transform, loader, loss, optimizer
    timm_name = M.resolve_name(cfg.backbone)
    model = M.build_model(timm_name, pretrained=cfg.pretrained, drop_rate=cfg.drop_rate, init=cfg.init,
                          drop_path_rate=cfg.drop_path_rate).to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    mean, std = _train_mean_std(model)
    tf_train = D.build_transforms(True, cfg.img_size, cfg.aug, mean, std)
    tf_eval = D.build_transforms(False, cfg.img_size, cfg.aug, mean, std)
    train_loader = D.make_loader(train_df, cfg.images_dir, tf_train, cfg.batch_size, True, cfg.sampler,
                                 cfg.num_workers, seed=cfg.seed, cache=cfg.cache_images)
    val_loader = D.make_loader(val_df, cfg.images_dir, tf_eval, cfg.batch_size, False, None,
                               cfg.num_workers, seed=cfg.seed, cache=cfg.cache_images)

    counts = D.class_counts(train_df)
    if cfg.loss == "ls":
        criterion = L.build_criterion("ls", smoothing=cfg.label_smoothing or 0.1)
    elif cfg.loss == "focal":
        criterion = L.build_criterion("focal", gamma=cfg.focal_gamma)
    elif cfg.loss == "ce_weighted":
        w = L.class_weights(counts, beta=cfg.class_weight_beta or 0.0)
        criterion = L.build_criterion("ce_weighted", weight=w)
    else:
        criterion = L.build_criterion("ce")
    criterion = criterion.to(device)
    eval_criterion = nn.CrossEntropyLoss().to(device)

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader) if cfg.limit_batches is None else min(len(train_loader), cfg.limit_batches)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = make_scaler(use_amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    eval_model = ema.module if ema is not None else model

    # 5) vòng epoch (có resume)
    history: list[dict] = []
    best = {"epoch": -1, "f1": -1.0}
    start = 0
    last_path = rd / "last.pt"
    if cfg.resume and last_path.exists():
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        if ema is not None and st.get("ema") is not None:
            ema.module.load_state_dict(st["ema"])
            ema.n = st["ema_n"]
        history, best, start = st["history"], st["best"], st["epoch"] + 1
        print(f"[{cfg.exp_id} seed{cfg.seed}] tiếp tục từ epoch {start} (không tái lập chính xác từng bit)")
    train_times = [h["train_time_s"] for h in history]

    for epoch in range(start, cfg.epochs):
        train_loader.generator.manual_seed(cfg.seed * 1000 + epoch)
        t0 = time.time()
        stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema, epoch)
        t_train = time.time() - t0
        t1 = time.time()
        names, y, logits, vloss = evaluate(eval_model, val_loader, eval_criterion, device, cfg.limit_batches)
        probs = softmax_np(logits)
        m = ev.compute_metrics(y, probs.argmax(1), probs)
        row = {"epoch": epoch, **stats, "val_loss": vloss, "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
               "val_balanced_acc": m["balanced_acc"], "val_ece": m["ece"],
               "train_time_s": t_train, "eval_time_s": time.time() - t1}
        row.update({f"val_f1_{i}": float(m["f1"][i]) for i in range(D.NUM_CLASSES)})
        row.update({f"val_recall_{i}": float(m["recall"][i]) for i in range(D.NUM_CLASSES)})
        if ema is not None:  # để so EMA với trọng số thô (I06)
            _, y2, lg2, _ = evaluate(model, val_loader, eval_criterion, device, cfg.limit_batches)
            p2 = softmax_np(lg2)
            row["val_macro_f1_raw"] = ev.compute_metrics(y2, p2.argmax(1), p2)["macro_f1"]
        history.append(row)
        train_times.append(t_train)
        print(f"[{cfg.exp_id} s{cfg.seed}] ep {epoch + 1}/{cfg.epochs} loss {stats['train_loss']:.4f} "
              f"val F1 {m['macro_f1']:.4f} top1 {m['top1']:.4f} ({t_train:.0f}s)")

        if m["macro_f1"] > best["f1"] + 1e-12:  # hòa thì giữ epoch sớm hơn
            best = {"epoch": epoch, "f1": float(m["macro_f1"])}
            if cfg.save_ckpt:
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in eval_model.state_dict().items()},
                            "epoch": epoch, "ema": ema is not None, "weight_tag": getattr(model, "weight_tag", "")},
                           rd / "best.pt")
            np.savez(rd / "val_logits.npz", files=np.array(names), y=y, logits=logits)
        pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
        if cfg.save_ckpt:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                        "history": history, "best": best,
                        "ema": ema.module.state_dict() if ema is not None else None,
                        "ema_n": ema.n if ema is not None else 0}, last_path)

    # 6) nạp best, ghi dự đoán val
    if cfg.save_ckpt and (rd / "best.pt").exists():
        eval_model.load_state_dict(torch.load(rd / "best.pt", map_location=device, weights_only=False)["state_dict"])
    z = np.load(rd / "val_logits.npz")
    if cfg.save_val_predictions:
        ev.save_predictions(pred_path(cfg, "val"), z["files"].tolist(), z["y"], softmax_np(z["logits"]))

    # 7) test (chỉ khi được bật; đường chính thức là final.py)
    if cfg.save_test_predictions:
        begin_test(rd, who="train.run")
        tf = D.build_transforms(False, cfg.img_size, cfg.aug, mean, std)
        test_loader = D.make_loader(test_df, cfg.images_dir, tf, cfg.batch_size, False, None, cfg.num_workers)
        tn, ty, tz, _ = evaluate(eval_model, test_loader, eval_criterion, device)
        np.savez(rd / "test_logits.npz", files=np.array(tn), y=ty, logits=tz)
        ev.save_predictions(pred_path(cfg, "test"), tn, ty, softmax_np(tz))
        finish_test(rd, {"n": int(len(ty)), "via": "train.run"})

    # 8) tổng kết
    h = pd.DataFrame(history)
    b = h.iloc[best["epoch"]]
    latency = None
    if cfg.measure_latency:
        try:
            latency = B.latency_report(eval_model, 1, cfg.img_size, "fp32", str(device), warmup=10, iters=50)
        except Exception as e:  # đo độ trễ không được làm hỏng cả lần chạy
            print("cảnh báo: không đo được độ trễ:", e)
    import timm
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "timm_name": timm_name,
        "weight_tag": getattr(model, "weight_tag", ""), "img_size": cfg.img_size, "epochs": cfg.epochs,
        "best_epoch": int(best["epoch"]) + 1,
        "val_macro_f1": float(b["val_macro_f1"]), "val_top1": float(b["val_top1"]),
        "val_balanced_acc": float(b["val_balanced_acc"]), "val_ece": float(b["val_ece"]),
        "val_f1_per_class": [float(b[f"val_f1_{i}"]) for i in range(D.NUM_CLASSES)],
        "val_recall_per_class": [float(b[f"val_recall_{i}"]) for i in range(D.NUM_CLASSES)],
        "val_macro_f1_raw": float(b["val_macro_f1_raw"]) if "val_macro_f1_raw" in b else None,
        "train_time_per_epoch_s": float(np.mean(train_times)),
        "params_M": M.count_params(model),
        "gmacs": M.count_gmacs(model, cfg.img_size),
        "latency_b1_ms": latency["p50"] if latency else None,
        "latency_b1_p95_ms": latency["p95"] if latency else None,
        "gpu": B.device_name(str(device)), "torch": torch.__version__, "timm": timm.__version__,
        "config": dataclasses.asdict(cfg),
    }
    (rd / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lr_steps = [lr_factor(s, cfg.epochs * steps_per_epoch, int(round(cfg.warmup_epochs * steps_per_epoch)))
                for s in range(cfg.epochs * steps_per_epoch)]
    plot_curves(history, curve_path(cfg), f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed} | {cfg.note or cfg.desc}",
                lr_steps)
    # dọn last.pt (to, chỉ dùng để resume)
    if last_path.exists():
        last_path.unlink()
    return summary


def alias_run(src_exp: str, dst_exp: str, seed: int, out_dir: str = "runs", curves_dir: str = "curves",
              desc: str = "", note: str = "") -> dict:
    """Dùng lại một lần chạy đã xong cho mã thí nghiệm khác khi CẤU HÌNH GIỐNG HỆT (ví dụ T00 = B0x của
    backbone được chọn, cùng seed): chép run_dir sang exp_id mới, ghi `alias_of` vào config/summary, vẽ lại ảnh.
    Tiết kiệm một lần huấn luyện; phải nêu rõ trong báo cáo (cột ghi chú của xlsx).
    """
    src = Path(out_dir) / src_exp / f"seed{seed}"
    dst = Path(out_dir) / dst_exp / f"seed{seed}"
    if not (src / "summary.json").exists():
        raise FileNotFoundError(f"{src} chưa chạy xong")
    if dst.exists():
        return json.loads((dst / "summary.json").read_text(encoding="utf-8"))
    shutil.copytree(src, dst)
    for f in ("test_started.json", "test_done.json", "test_logits.npz", "last.pt"):
        if (dst / f).exists():
            (dst / f).unlink()
    cfgd = json.loads((dst / "config.json").read_text(encoding="utf-8"))
    cfgd.update({"exp_id": dst_exp, "alias_of": src_exp, "desc": desc or cfgd.get("desc", ""),
                 "note": note or cfgd.get("note", "")})
    (dst / "config.json").write_text(json.dumps(cfgd, indent=2, ensure_ascii=False), encoding="utf-8")
    s = json.loads((dst / "summary.json").read_text(encoding="utf-8"))
    s.update({"exp_id": dst_exp, "alias_of": src_exp})
    s["config"] = cfgd
    (dst / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    hist = pd.read_csv(dst / "history.csv")
    cfg = Config(**{k: v for k, v in cfgd.items() if k in Config.__dataclass_fields__})
    cfg.curves_dir = curves_dir
    spe = None
    plot_curves(hist.to_dict("records"), curve_path(cfg),
                f"{dst_exp} (= {src_exp}) | {cfg.backbone} | seed {seed} | {cfg.note or cfg.desc}", spe)
    return s


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    fields = {f.name: f for f in dataclasses.fields(Config)}
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"'{p}' phải có dạng KEY=VALUE")
        k, v = p.split("=", 1)
        if k not in fields:
            raise ValueError(f"'{k}' không có trong Config. Các key hợp lệ: {sorted(fields)}")
        t = str(fields[k].type)
        if "None" in t and v.lower() in ("none", "null"):
            out[k] = None
        elif t.startswith("bool"):
            if v.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"{k} cần true/false, nhận '{v}'")
            out[k] = v.lower() in ("true", "1")
        elif t.startswith("int"):
            out[k] = int(v)
        elif t.startswith("float"):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình (Lab Day 2)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ghi đè field của Config")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    s = run(cfg)
    print(json.dumps({k: s[k] for k in ("exp_id", "seed", "best_epoch", "val_macro_f1", "val_top1",
                                         "train_time_per_epoch_s", "params_M", "gmacs")}, indent=2))


if __name__ == "__main__":
    main()
