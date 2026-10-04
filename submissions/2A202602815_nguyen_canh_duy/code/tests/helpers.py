"""Tiện ích test: sinh bộ dữ liệu giả có cùng cấu trúc DeepWeeds (CSV + ảnh 256x256), chạy được trên CPU."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CODE = Path(__file__).resolve().parent.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

SPECIES = ["Chinee apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly acacia",
           "Rubber vine", "Siam weed", "Snake weed", "Negative"]


def make_fake_deepweeds(root: Path, per_class=(12, 4, 4), size: int = 256, seed: int = 0):
    """Tạo root/images/*.jpg và root/labels/{labels,train_subset0,val_subset0,test_subset0}.csv (tỉ lệ 60/20/20)."""
    rng = np.random.default_rng(seed)
    img_dir, lab_dir = root / "images", root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "val": [], "test": []}
    k = 0
    for c in range(9):
        base = np.array([(c * 28) % 256, (c * 71) % 256, (c * 113) % 256], dtype=np.float64)
        for split, n in zip(("train", "val", "test"), per_class):
            for _ in range(n):
                arr = np.clip(base + rng.normal(0, 25, (size, size, 3)), 0, 255).astype(np.uint8)
                name = f"fake-{k:05d}.jpg"
                Image.fromarray(arr).save(img_dir / name, quality=90)
                rows[split].append({"Filename": name, "Label": c, "Species": SPECIES[c]})
                k += 1
    dfs = {s: pd.DataFrame(r) for s, r in rows.items()}
    for s, df in dfs.items():
        df.to_csv(lab_dir / f"{s}_subset0.csv", index=False)
    pd.concat(dfs.values()).to_csv(lab_dir / "labels.csv", index=False)
    return img_dir, lab_dir, dfs
