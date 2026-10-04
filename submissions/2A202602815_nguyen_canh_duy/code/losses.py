"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện (giữ đúng như bộ khung starter):
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Tự cài đặt (không dùng CrossEntropyLoss(label_smoothing=...)) và kiểm tra khớp bản của PyTorch
    trong tests. eps = 0 cho đúng CE.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải trong [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
        uniform = -logp.mean(dim=-1)  # = sum_k (1/K) * (-log p_k)
        return ((1.0 - self.smoothing) * nll + self.smoothing * uniform).mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    gamma = 0 và alpha = None cho đúng cross-entropy (kiểm tra trong tests, sai số < 1e-6).
    alpha: None hoặc vector trọng số theo lớp (độ dài K).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=-1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


class _WeightedCE(nn.Module):
    """CE có trọng số lớp. Dùng trung bình đơn giản trên batch (không chia cho tổng trọng số như
    nn.CrossEntropyLoss(weight=...)), để thang loss không đổi theo thành phần lớp của batch."""

    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32))

    def forward(self, logits, target):
        return (F.cross_entropy(logits.float(), target, reduction="none") * self.weight[target]).mean()


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor.
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("ce_weighted cần weight=class_weights(counts, beta)")
        return _WeightedCE(kw["weight"])
    raise ValueError(f"loss={kind!r} không hợp lệ (ce | ls | focal | ce_weighted)")


def class_weights(counts, beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN (không dùng val hay test).

    - beta = 0: w_c = 1 / n_c, chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c),
      chuẩn hoá tổng trọng số về số lớp (Cui et al., arXiv:1901.05555; slide trang 57)
    """
    n = np.asarray(counts, dtype=np.float64)
    if (n <= 0).any():
        raise ValueError("mọi lớp phải có ít nhất 1 ảnh trong train")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta phải trong [0, 1)")
    w = 1.0 / n if beta == 0.0 else (1.0 - beta) / (1.0 - np.power(beta, n))
    w = w / w.sum() * len(n)
    return torch.as_tensor(w, dtype=torch.float32)


def _rand_bbox(h: int, w: int, lam: float, rng) -> tuple[int, int, int, int]:
    cut = np.sqrt(1.0 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = int(rng.integers(h)), int(rng.integers(w))
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    return int(y1), int(y2), int(x1), int(x2)


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix", rng=None):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp từ x[perm] dán vào x, rồi đặt lại lam theo DIỆN TÍCH THỰC của hộp
      sau khi cắt ra ngoài biên (slide trang 48)
    - trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]; không sửa x tại chỗ.
    `rng`: numpy Generator (mặc định lấy từ numpy global đã gieo bằng set_seed).
    """
    if mode not in ("mixup", "cutmix"):
        raise ValueError("mode phải là 'mixup' hoặc 'cutmix'")
    rng = rng if rng is not None else np.random.default_rng(np.random.randint(0, 2 ** 31 - 1))
    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    else:
        h, w = x.shape[-2:]
        y1, y2, x1, x2 = _rand_bbox(h, w, lam, rng)
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    Accuracy trên batch đã trộn không còn nghĩa bình thường; đánh giá bằng val.
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
