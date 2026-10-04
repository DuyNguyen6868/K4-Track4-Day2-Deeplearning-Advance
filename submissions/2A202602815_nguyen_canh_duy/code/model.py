"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện (giữ đúng như bộ khung starter):
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

# Mọi backbone dùng bộ trọng số CHỈ HUẤN LUYỆN TRÊN ImageNet-1k, ghim tag tường minh để so sánh công bằng.
# Không dùng tag mặc định: ví dụ `convnext_tiny` mặc định của timm hiện là `in12k_ft_in1k` (tiền huấn luyện
# thêm trên ImageNet-12k), sẽ làm so sánh với ResNet lệch về dữ liệu tiền huấn luyện.
# Lưu ý: ResNet-50/ResNeXt-50 dùng công thức "ResNet strikes back" (a1/a1h), mạnh hơn công thức gốc của torchvision;
# đây là điểm cần nêu khi bàn "kiến trúc hay công thức" (GUIDE mục 9, câu 1).
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50.a1_in1k",
    "resnext50": "resnext50_32x4d.a1h_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "deit_small": "deit_small_patch16_224.fb_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "mobilenetv3": "mobilenetv3_large_100.ra_in1k",
}

INITS = ("scratch", "frozen", "finetune")


def resolve_name(name: str) -> str:
    """Khoá ngắn trong SUGGESTED_BACKBONES -> tên timm có ghim tag; tên khác giữ nguyên (ví dụ resnet18)."""
    return SUGGESTED_BACKBONES.get(name, name)


def weight_tag(model) -> str:
    """Tên + tag trọng số thực sự được dùng, để ghi vào results.xlsx."""
    cfg = getattr(model, "pretrained_cfg", None) or {}
    arch = cfg.get("architecture") or getattr(model, "default_cfg", {}).get("architecture", "")
    tag = cfg.get("tag") or ""
    hub = cfg.get("hf_hub_id") or cfg.get("url") or ""
    return f"{arch}.{tag}".strip(".") + (f" [{hub}]" if hub else "")


def build_model(name: str, pretrained: bool = True, num_classes: int = 9, drop_rate: float = 0.0,
                init: str = "finetune", drop_path_rate: float = 0.0):
    """Tạo model phân loại 9 lớp bằng timm (timm tự thay head mới, khởi tạo ngẫu nhiên).

    `init`: "scratch" (pretrained=False, train toàn bộ), "frozen" (pretrained, đóng băng backbone, chỉ train
    head), "finetune" (pretrained, train toàn bộ). `pretrained=False` với finetune/frozen chỉ dùng để chạy thử
    offline. Gắn `model.weight_tag` (tag trọng số) và `model.frozen_backbone` (bool).
    """
    if init not in INITS:
        raise ValueError(f"init={init!r} không hợp lệ, chọn một trong {INITS}")
    use_pretrained = pretrained and init != "scratch"
    kw = dict(pretrained=use_pretrained, num_classes=num_classes, drop_rate=drop_rate)
    if drop_path_rate:
        kw["drop_path_rate"] = drop_path_rate
    model = timm.create_model(name, **kw)
    model.weight_tag = weight_tag(model) if use_pretrained else "scratch (không tiền huấn luyện)"
    model.frozen_backbone = False
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head (model.get_classifier()).

    Backbone đóng băng thì BatchNorm cũng phải ở chế độ eval, nếu không thống kê BN vẫn bị cập nhật theo
    batch train trong khi trọng số đứng yên. Vòng train gọi `set_train_mode(model)` thay cho model.train().
    """
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    model.frozen_backbone = True


def set_train_mode(model) -> None:
    """model.train(), nhưng nếu backbone đóng băng thì cả mạng ở eval và chỉ head ở train."""
    model.train()
    if getattr(model, "frozen_backbone", False):
        model.eval()
        model.get_classifier().train()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1) cùng các tham số trong model.no_weight_decay() (pos_embed,
      cls_token, bảng bias tương đối của Swin...): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay
    Bỏ qua tham số requires_grad == False; nhóm rỗng bị loại.
    """
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    skip = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    decay, no_decay, head = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head.append(p)
        elif p.ndim <= 1 or name in skip or any(name.endswith("." + s) for s in skip):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"},
        {"params": no_decay, "lr": lr_backbone, "weight_decay": 0.0, "name": "backbone_no_decay"},
        {"params": head, "lr": lr_head, "weight_decay": weight_decay, "name": "head"},
    ]
    return [g for g in groups if g["params"]]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (MAC, không phải FLOPs gấp đôi).

    Công cụ: torch.utils.flop_counter.FlopCounterMode (đếm conv, matmul, bmm, attention) rồi chia 2.
    Chạy trên CPU với bản sao của model; số có thể lệch vài phần trăm so với fvcore/ptflops (không đếm
    các phép phần tử như BN, activation, softmax).
    """
    import copy
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, 3, img_size, img_size)
    with FlopCounterMode(display=False) as counter:
        m(x)
    return counter.get_total_flops() / 2 / 1e9
