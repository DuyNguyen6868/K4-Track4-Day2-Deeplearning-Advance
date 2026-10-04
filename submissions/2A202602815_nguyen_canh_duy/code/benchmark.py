"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch

Quy ước của bài này: độ trễ chỉ gồm lượt forward của model trên tensor đã nằm sẵn trên thiết bị
(KHÔNG gồm giải mã JPEG, tiền xử lý, chép host->device). Ghi điều này vào báo cáo.
"""
from __future__ import annotations

import copy
import platform
import time

import numpy as np
import torch

MIN_WARMUP = 10
MIN_ITERS = 50


def bench(fn, warmup: int = MIN_WARMUP, iters: int = 100, sync=None, strict: bool = True) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.
    strict=True ép warmup >= 10 và iters >= 50 theo quy định của bài lab (tắt chỉ khi chạy thử).
    """
    if strict and (warmup < MIN_WARMUP or iters < MIN_ITERS):
        raise ValueError(f"cần warmup >= {MIN_WARMUP} và iters >= {MIN_ITERS} (nhận {warmup}, {iters})")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times[i] = (time.perf_counter() - t0) * 1000.0
    return {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)), "mean": float(times.mean()), "n": int(iters)}


def device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.device(device))
    return f"CPU ({platform.processor() or platform.machine()})"


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = MIN_WARMUP, iters: int = 100, fused_bn: bool = False,
                   strict: bool = True) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast fp16) | "fp16" (model.half()). `fused_bn` chỉ để ghi chú vào kết quả
    (truyền vào model đã gộp BN bằng inference.fuse_conv_bn). Trả về dict ghi thẳng vào sheet `Latency`.
    Ở batch 1 AMP có thể CHẬM hơn FP32 (slide trang 73): đo thật, đừng giả định.
    """
    if dtype not in ("fp32", "amp", "fp16"):
        raise ValueError("dtype phải là fp32 | amp | fp16")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    m = copy.deepcopy(model).to(dev).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        m = m.half()
        x = x.half()
    use_cuda = dev.type == "cuda"
    sync = torch.cuda.synchronize if use_cuda else None
    amp_dtype = torch.float16 if use_cuda else torch.bfloat16

    @torch.inference_mode()
    def fn():
        if dtype == "amp":
            with torch.autocast(device_type=dev.type, dtype=amp_dtype):
                m(x)
        else:
            m(x)

    r = bench(fn, warmup=warmup, iters=iters, sync=sync, strict=strict)
    return {"gpu": device_name(device), "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "fused_bn": bool(fused_bn), "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "mean": r["mean"],
            "n": r["n"], "warmup": warmup, "images_per_s": batch_size / (r["p50"] / 1000.0),
            "torch": torch.__version__, "includes_preprocessing": False}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view chạy tuần tự (mỗi view một lượt forward, batch như `kw`).

    Trả về dict như latency_report kèm `single_p50`, `expected_p50` (= K * p50 một view) và
    `ratio_vs_single` để so với kỳ vọng "chi phí gần tuyến tính theo K" (slide trang 63).
    """
    single = latency_report(model, **kw)
    batch_size, img_size = kw["batch_size"], kw["img_size"]
    device = kw.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dtype = kw.get("dtype", "fp32")
    dev = torch.device(device)
    m = copy.deepcopy(model).to(dev).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        m, x = m.half(), x.half()
    use_cuda = dev.type == "cuda"

    @torch.inference_mode()
    def fn():
        for _ in range(k_views):
            if dtype == "amp":
                with torch.autocast(device_type=dev.type, dtype=torch.float16 if use_cuda else torch.bfloat16):
                    m(x)
            else:
                m(x)

    r = bench(fn, warmup=kw.get("warmup", MIN_WARMUP), iters=kw.get("iters", 100),
              sync=torch.cuda.synchronize if use_cuda else None, strict=kw.get("strict", True))
    out = dict(single)
    out.update({"k_views": k_views, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "mean": r["mean"],
                "single_p50": single["p50"], "expected_p50": k_views * single["p50"],
                "ratio_vs_single": r["p50"] / single["p50"],
                "images_per_s": batch_size / (r["p50"] / 1000.0)})
    return out
