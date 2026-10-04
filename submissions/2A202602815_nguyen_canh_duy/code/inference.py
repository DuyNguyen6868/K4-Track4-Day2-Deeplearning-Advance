"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val; nhiệt độ T khớp trên
VAL rồi áp dụng sang test (README.md, S2 và S4).

Thiết kế "một lượt đọc ảnh, nhiều view": các view đều sinh ra từ ẢNH ĐẦY ĐỦ 256x256 (loader dùng
`build_transforms(train=False, test_size=256)`), nên view "center" (cắt giữa 224) trùng đúng với mốc I00.
Nhờ vậy tập test chỉ cần đi qua loader MỘT lần cho mọi view (GUIDE mục 5).

Giao diện (giữ như starter):
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Các view (mỗi view: hàm batch (N,C,H,W) -> batch)
# --------------------------------------------------------------------------- #
def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def _crop(x, top: int, left: int, size: int):
    return x[..., top:top + size, left:left + size]


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop`, tuỳ chọn thêm bản lật của từng crop. Trả về list batch."""
    h, w = x.shape[-2:]
    if crop > min(h, w):
        raise ValueError(f"crop={crop} lớn hơn ảnh {h}x{w}")
    tops_lefts = [((h - crop) // 2, (w - crop) // 2), (0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop)]
    out = [_crop(x, t, l, crop) for t, l in tops_lefts]
    if flip:
        out += [torch.flip(v, dims=[-1]) for v in out]
    return out


def views_multiscale(x, sizes):
    """Resize cả ảnh về từng kích thước trong `sizes` (nội suy song tuyến, antialias), trả về list batch.

    Chỉ dùng cho CNN có global pooling; ViT/Swin có kích thước đầu vào cố định (ghi rõ giới hạn trong báo cáo).
    """
    return [x if x.shape[-1] == s and x.shape[-2] == s else
            F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=True) for s in sizes]


def standard_views(crop: int = 224, sizes=(224, 256, 288, 320)) -> dict:
    """Từ điển tên view -> hàm. Đầu vào của mọi view là batch ảnh đầy đủ (N, 3, 256, 256)."""
    def center(x):
        return views_multicrop(x, crop)[0]

    def make_crop(i):
        return lambda x: views_multicrop(x, crop)[i]

    views = {"center": center, "center_flip": lambda x: torch.flip(center(x), dims=[-1])}
    for i, nm in enumerate(["tl", "tr", "bl", "br"], start=1):
        views[nm] = make_crop(i)
        views[nm + "_flip"] = (lambda j: (lambda x: torch.flip(views_multicrop(x, crop)[j], dims=[-1])))(i)
    for s in sizes:
        views[f"full{s}"] = (lambda size: (lambda x: views_multiscale(x, [size])[0]))(s)
    return views


# Các bộ view dùng ở Bước 3/4 (tên view phải có trong standard_views).
VIEW_SETS = {
    "I00": ["center"],
    "I01": ["center", "center_flip"],
    "I02a": ["center", "tl", "tr", "bl", "br"],
    "I02b": ["center", "tl", "tr", "bl", "br", "center_flip", "tl_flip", "tr_flip", "bl_flip", "br_flip"],
    "I02c": ["full224", "full256", "full288"],
}


@torch.inference_mode()
def predict_views(model, loader, device, views: dict, dtype: str = "fp32", channels_last: bool = False):
    """Chạy `loader` MỘT lần, mỗi batch qua mọi view. Trả về (filenames, y_true, {tên_view: logits[N, 9]}).

    dtype: "fp32" | "amp" | "fp16" (model sẽ bị đổi sang half tại chỗ nếu fp16; truyền bản sao nếu cần).
    """
    device = torch.device(device)
    model.eval().to(device)
    if dtype == "fp16":
        model.half()
    names, ys = [], []
    outs = {k: [] for k in views}
    for x, y, f in loader:
        x = x.to(device, non_blocking=True)
        for k, fn in views.items():
            v = fn(x)
            if channels_last:
                v = v.contiguous(memory_format=torch.channels_last)
            if dtype == "fp16":
                v = v.half()
            if dtype == "amp":
                with torch.autocast(device_type=device.type,
                                    dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                    z = model(v)
            else:
                z = model(v)
            outs[k].append(z.float().cpu().numpy())
        ys.append(np.asarray(y))
        names.extend(f)
    return names, np.concatenate(ys), {k: np.concatenate(v) for k, v in outs.items()}


def predict_logits(model, loader, device, view=None, dtype: str = "fp32"):
    """Chạy model trên loader và gom logit theo đúng thứ tự file. `view`: hàm biến đổi batch hoặc None."""
    names, y, out = predict_views(model, loader, device, {"v": view or view_identity}, dtype=dtype)
    return names, y, out["v"]


# --------------------------------------------------------------------------- #
# Gộp, ensemble, hiệu chuẩn
# --------------------------------------------------------------------------- #
def softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành xác suất (N, 9) (slide trang 62).

    space="prob": trung bình softmax của từng view; space="logit": trung bình logit rồi softmax.
    """
    if space not in ("prob", "logit"):
        raise ValueError("space phải là 'prob' hoặc 'logit'")
    arr = np.stack([np.asarray(z, dtype=np.float64) for z in logits_per_view])
    if space == "prob":
        return softmax_np(arr, axis=-1).mean(axis=0)
    return softmax_np(arr.mean(axis=0))


def ensemble_probs(list_of_probs) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file."""
    arrs = [np.asarray(p, dtype=np.float64) for p in list_of_probs]
    if len({a.shape for a in arrs}) != 1:
        raise ValueError("các mô hình phải cùng số ảnh và số lớp")
    return np.mean(arrs, axis=0)


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())


def _minimize_log_t(f, lo: float = np.log(0.05), hi: float = np.log(20.0)) -> float:
    """Cực tiểu f(log T): lưới thô 200 điểm rồi tìm vàng (golden section) quanh điểm tốt nhất."""
    grid = np.linspace(lo, hi, 201)
    vals = [f(g) for g in grid]
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    phi = (np.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return float(np.exp((a + b) / 2))


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T) (slide trang 69).

    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    z = np.asarray(val_logits, dtype=np.float64)
    y = np.asarray(val_labels, dtype=np.int64)
    return _minimize_log_t(lambda lt: _nll(softmax_np(z / np.exp(lt)), y))


def fit_temperature_views(val_logits_per_view, val_labels, space: str = "prob") -> float:
    """Như fit_temperature nhưng cho TTA: T áp dụng lên logit từng view TRƯỚC khi gộp (space prob/logit)."""
    y = np.asarray(val_labels, dtype=np.int64)
    return _minimize_log_t(lambda lt: _nll(aggregate_views(
        [z / np.exp(lt) for z in val_logits_per_view], space), y))


def apply_temperature(logits, T: float) -> np.ndarray:
    """softmax(logits / T)."""
    return softmax_np(np.asarray(logits, dtype=np.float64) / T)


# --------------------------------------------------------------------------- #
# Gộp BatchNorm vào conv
# --------------------------------------------------------------------------- #
def fuse_conv_bn(model, check_size: int = 64, verbose: bool = True):
    """Trả về BẢN SAO của model với BatchNorm2d gộp vào Conv2d liền trước, chính xác lúc suy luận:

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Quy tắc ghép: trong cùng một module cha, Conv2d đứng ngay trước BatchNorm2d theo thứ tự đăng ký
    (đúng với ResNet/ResNeXt/Sequential của timm). BatchNormAct2d (EfficientNet, MobileNetV3) được thay bằng
    phần activation của nó. Sau khi gộp, so đầu ra trước/sau trên một batch ngẫu nhiên và in sai số lớn nhất;
    nếu sai số > 1e-3 thì ném lỗi (ghép sai cặp). Kiến trúc không có BN (ViT, Swin, ConvNeXt) trả về
    bản sao nguyên vẹn với `fuse_info["n_fused"] == 0`.
    """
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    ref = model.eval()
    fused = copy.deepcopy(model).eval()
    n = 0
    for parent in list(fused.modules()):
        kids = list(parent.named_children())
        for (n1, m1), (n2, m2) in zip(kids, kids[1:]):
            if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d):
                setattr(parent, n1, fuse_conv_bn_eval(m1, m2))
                if hasattr(m2, "act") and hasattr(m2, "drop"):
                    setattr(parent, n2, nn.Sequential(m2.drop, m2.act))
                else:
                    setattr(parent, n2, nn.Identity())
                n += 1
    dev = next(fused.parameters()).device
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    ch = (cfg.get("input_size") or (3, check_size, check_size))[0]
    x = torch.randn(2, ch, check_size, check_size, device=dev)
    with torch.inference_mode():
        try:
            err = float((ref(x) - fused(x)).abs().max())
        except Exception:  # kiến trúc đòi kích thước cố định (ViT/Swin): thử lại bằng kích thước mặc định
            size = (cfg.get("input_size") or (3, 224, 224))[1]
            x = torch.randn(2, ch, size, size, device=dev)
            err = float((ref(x) - fused(x)).abs().max())
    fused.fuse_info = {"n_fused": n, "max_abs_err": err}
    if verbose:
        print(f"fuse_conv_bn: gộp {n} cặp conv+BN, sai số đầu ra lớn nhất = {err:.2e}"
              + ("  (kiến trúc không có BN: không áp dụng)" if n == 0 else ""))
    if err > 1e-3:
        raise RuntimeError(f"sai số sau khi gộp BN quá lớn ({err:.2e}); kiểm tra lại cách ghép cặp")
    return fused
