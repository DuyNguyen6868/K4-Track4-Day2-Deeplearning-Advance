"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu (S1-S6, README.md mục 2.1): dùng đúng fold 0 do tác giả chia sẵn, không sửa,
không lọc, không chia lại, không gộp val vào train.

Giao diện (giữ đúng như bộ khung starter):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
EXPECTED_TOTAL = 17509
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Số ảnh theo lớp ở Table 1 của bài báo (dùng để đối chiếu khi EDA, không dùng để huấn luyện).
PAPER_COUNTS = {"Chinee Apple": 1125, "Lantana": 1064, "Parkinsonia": 1031, "Parthenium": 1022,
                "Prickly Acacia": 1062, "Rubber Vine": 1009, "Siam Weed": 1074, "Snake Weed": 1016,
                "Negatives": 9106}
AUG_CHOICES = ("basic", "mildcrop", "geo", "color", "trivial", "randaug")


# --------------------------------------------------------------------------- #
# Đọc và kiểm tra chia dữ liệu
# --------------------------------------------------------------------------- #
def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train/val/test_subset{fold}.csv (S1). Không sửa, không lọc, không chia lại."""
    d = Path(labels_dir)
    out = []
    for name in ("train", "val", "test"):
        path = d / f"{name}_subset{fold}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu {path}. Tải nguyên bản từ github.com/AlexOlsen/DeepWeeds/labels")
        df = pd.read_csv(path)
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"{path}: thiếu cột {sorted(missing)}")
        out.append(df)
    return tuple(out)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, expected_total: int | None = EXPECTED_TOTAL,
                max_ratio_dev: float = 0.01, check_files: bool = True, verbose: bool = True) -> dict:
    """Các kiểm tra bắt buộc trước khi train (README.md mục 2.1). In ra và trả về dict số liệu.

    Dừng (raise ValueError) nếu: giao của hai tập khác rỗng, tên file trùng trong một tập, hợp ba tập khác
    `expected_total`, tỉ lệ lệch 60/20/20 quá `max_ratio_dev`, hoặc có file không tồn tại.
    """
    sets = {"train": train_df, "val": val_df, "test": test_df}
    names = {k: set(v["Filename"]) for k, v in sets.items()}
    for k, v in sets.items():
        if v["Filename"].duplicated().any():
            raise ValueError(f"{k}: có Filename bị trùng")

    n = {k: int(len(v)) for k, v in sets.items()}
    total = sum(n.values())
    per_class = {k: {CLASS_NAMES[c]: int((v["Label"] == c).sum()) for c in range(NUM_CLASSES)}
                 for k, v in sets.items()}
    overlap = {"train&val": len(names["train"] & names["val"]),
               "train&test": len(names["train"] & names["test"]),
               "val&test": len(names["val"] & names["test"])}
    union = len(names["train"] | names["val"] | names["test"])
    ratio = {k: n[k] / total for k in n}

    missing_files = 0
    if check_files:
        root = Path(images_dir)
        missing_files = sum(1 for f in (names["train"] | names["val"] | names["test"]) if not (root / f).is_file())

    result = {"n": n, "total": total, "ratio": ratio, "per_class": per_class, "overlap": overlap,
              "union": union, "missing_files": missing_files}

    if verbose:
        print("Số ảnh:", n, "| tổng", total, "| tỉ lệ", {k: f"{v:.3%}" for k, v in ratio.items()})
        print(pd.DataFrame(per_class).assign(tổng=lambda d: d.sum(axis=1)).to_string())
        print("Giao các tập:", overlap, "| hợp:", union, "| file thiếu:", missing_files)

    problems = []
    if any(overlap.values()):
        problems.append(f"giao các tập khác rỗng: {overlap}")
    if expected_total is not None and union != expected_total:
        problems.append(f"hợp ba tập = {union}, kỳ vọng {expected_total}")
    for k, target in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        if abs(ratio[k] - target) > max_ratio_dev:
            problems.append(f"tỉ lệ {k} = {ratio[k]:.3%} lệch {target:.0%} quá {max_ratio_dev:.0%}")
    if missing_files:
        problems.append(f"{missing_files} file trong CSV không tồn tại trong {images_dir}")
    if problems:
        raise ValueError("Kiểm tra chia dữ liệu THẤT BẠI (báo giảng viên trước khi chạy tiếp): " + "; ".join(problems))
    return result


def class_counts(df: pd.DataFrame) -> np.ndarray:
    """Số ảnh mỗi lớp (độ dài 9) của một DataFrame; dùng cho trọng số lớp và sampler (chỉ train)."""
    return np.bincount(df["Label"].to_numpy(dtype=np.int64), minlength=NUM_CLASSES)


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD, test_size: int | None = None):
    """Tạo transform.

    Train (mọi mức): RandomResizedCrop(img_size) + lật ngang, rồi thêm theo `aug`:
        basic    : không thêm
        mildcrop : RandomResizedCrop scale (0.35, 1) thay cho (0.08, 1) - cắt nhẹ hơn, đỡ mất vật thể nhỏ
        geo      : + lật dọc (ảnh chụp từ trên xuống, hướng không mang nghĩa)
        color    : + ColorJitter (độ sáng, tương phản, bão hoà, sắc độ nhẹ)
        trivial  : + TrivialAugmentWide
        randaug  : + RandAugment(2, 9)
    Mixup/CutMix trộn theo batch nên nằm ở losses.py.

    Val/test (không ngẫu nhiên):
        test_size=None: Resize(max(256, img_size)) rồi CenterCrop(img_size). Ảnh gốc 256x256 nên với
                        img_size=224 đây là cắt giữa 224 từ ảnh 256 (I00 mốc).
        test_size=S   : Resize((S, S)), toàn bộ ảnh, không cắt (dùng cho TTA/độ phân giải kiểm tra).
    `mean`/`std` nên lấy từ `model.pretrained_cfg` (ViT augreg dùng 0.5/0.5, không phải ImageNet).
    """
    if aug not in AUG_CHOICES:
        raise ValueError(f"aug={aug!r} không hợp lệ, chọn một trong {AUG_CHOICES}")
    norm = [T.ToTensor(), T.Normalize(mean=list(mean), std=list(std))]
    if not train:
        if test_size is not None:
            return T.Compose([T.Resize((test_size, test_size)), *norm])
        return T.Compose([T.Resize(max(256, img_size)), T.CenterCrop(img_size), *norm])

    scale = (0.35, 1.0) if aug == "mildcrop" else (0.08, 1.0)
    ops = [T.RandomResizedCrop(img_size, scale=scale), T.RandomHorizontalFlip()]
    if aug == "geo":
        ops.append(T.RandomVerticalFlip())
    elif aug == "color":
        ops.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    return T.Compose([*ops, *norm])


def denormalize(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """Giải chuẩn hoá (C,H,W) hoặc (N,C,H,W) về [0,1] để vẽ ảnh kiểm tra."""
    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    s = torch.tensor(std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    return (x * s + m).clamp(0, 1)


# --------------------------------------------------------------------------- #
# Dataset và DataLoader
# --------------------------------------------------------------------------- #
class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) -> (ảnh đã transform, nhãn int, tên file str). Tên file cần có để ghi predictions/*.csv.
    cache=True nạp trước toàn bộ ảnh vào RAM (ảnh 256x256x3 uint8, 17,5k ảnh ~ 3,4 GB) nếu nghẽn đọc đĩa.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None, cache: bool = False):
        self.files = df["Filename"].tolist()
        self.labels = df["Label"].to_numpy(dtype=np.int64)
        self.root = Path(images_dir)
        self.transform = transform
        self._cache = None
        if cache:
            self._cache = [np.asarray(self._open(f)) for f in self.files]

    def _open(self, name: str) -> Image.Image:
        with Image.open(self.root / name) as im:
            return im.convert("RGB")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        img = Image.fromarray(self._cache[i]) if self._cache is not None else self._open(self.files[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), self.files[i]


def _worker_init(worker_id: int) -> None:
    """Gieo seed cho numpy/random trong worker (torch tự gieo riêng cho mỗi worker)."""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0, cache: bool = False) -> DataLoader:
    """Tạo DataLoader.

    train=True: shuffle (hoặc sampler), drop_last=True (BatchNorm không ổn định với batch cuối quá nhỏ).
    train=False: không shuffle, giữ đúng thứ tự `df` (thứ tự phải ổn định để ghép logit với Filename).
    sampler=None | "balanced": "balanced" dùng WeightedRandomSampler, trọng số 1/(số ảnh của lớp), lấy len(df)
    mẫu có hoàn lại mỗi epoch.
    Seed của generator có thể đặt lại mỗi epoch qua `loader.generator.manual_seed(...)`.
    """
    if sampler not in (None, "balanced"):
        raise ValueError(f"sampler={sampler!r} không hợp lệ (None | 'balanced')")
    ds = DeepWeedsDataset(df, images_dir, transform, cache=cache)
    g = torch.Generator()
    g.manual_seed(seed)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              worker_init_fn=_worker_init, generator=g)
    if num_workers > 0:
        kw["persistent_workers"] = train
    if train:
        if sampler == "balanced":
            counts = class_counts(df).astype(np.float64)
            w = 1.0 / np.maximum(counts, 1)
            weights = torch.as_tensor(w[ds.labels], dtype=torch.double)
            sm = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True, generator=g)
            return DataLoader(ds, sampler=sm, drop_last=True, **kw)
        return DataLoader(ds, shuffle=True, drop_last=True, **kw)
    return DataLoader(ds, shuffle=False, drop_last=False, **kw)


def save_check(result: dict, path: str | Path) -> None:
    """Lưu kết quả check_split ra JSON để dán vào báo cáo."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
