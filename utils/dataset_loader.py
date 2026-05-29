###  utils/dataset_loader.py

from __future__ import annotations

import os, glob, random
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler
from torchvision import transforms as T
import torchvision.transforms.functional as TF


# =========================
# Constants
# =========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Keep WEATHER2ID only for severity logic 
WEATHER2ID = {
    "clear": 0, "snow": 1, "rain": 2, "fog": 3, "night": 4, "haze": 5, "low": 6
}

# CDD-11 composite folders map to a *base* weather type for severity estimation (COARSE)
CDD11_FOLDER2WEATHER = {
    "clear": "clear",
    "snow": "snow",
    "rain": "rain",
    "haze": "haze",
    "low": "low",
    # Composite mapping
    "low_haze": "low", "low_rain": "low", "low_snow": "low",
    "haze_rain": "rain", "haze_snow": "snow",
    "low_haze_rain": "low", "low_haze_snow": "low",
}

# CDD-11 only classes (12)
STYLE_LABELS = [
    "clear", "haze", "haze_rain", "haze_snow",
    "low", "low_haze", "low_haze_rain", "low_haze_snow",
    "low_rain", "low_snow", "rain", "snow",
]
STYLE2ID = {name: idx for idx, name in enumerate(STYLE_LABELS)}


# =========================
# Utility functions
# =========================
def _maybe_joint_flip(x, y=None):
    if random.random() < 0.5:
        x = TF.hflip(x)
        y = TF.hflip(y) if y is not None else None
    return x, y


@torch.no_grad()
def estimate_severity(img_tensor: torch.Tensor, weather: str) -> float:
    """
    Cheap severity proxy used by your code.
    """
    g = img_tensor.mean(0)
    mean_lum = float(g.mean().item())
    contrast = float(g.std().item())

    if weather in ("night", "low"):
        sev = 1.0 - mean_lum
    elif weather in ("fog", "haze"):
        sev = 0.5 - (contrast - 0.10)
    elif weather in ("rain", "snow"):
        sev = 0.6 - (contrast - 0.12) + (0.5 - mean_lum)
    else:
        sev = 0.0

    return float(max(0.0, min(1.0, sev)))


def collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            if any(v is None for v in vals):
                out[k] = vals
            else:
                out[k] = torch.stack(vals, 0)
        else:
            out[k] = vals
    return out


def _maybe_build_sampler(dataset, distributed: bool, shuffle: bool):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
    return RandomSampler(dataset) if shuffle else SequentialSampler(dataset)


def make_cdd11_split(root: str, out_dir: Optional[str] = None, val_ratio: float = 0.1, seed: int = 42):
    """
    Builds:
      splits/cdd11_train.txt
      splits/cdd11_val.txt
    from paired images in:
      <root>/train/clear
      <root>/train/<style>
    (where style is any folder except 'clear')
    """
    base = os.path.abspath(root)
    if out_dir is None:
        out_dir = os.path.join(base, "splits")
    os.makedirs(out_dir, exist_ok=True)

    clear_dir  = os.path.join(base, "train", "clear")
    train_root = os.path.join(base, "train")

    if not (os.path.isdir(clear_dir) and os.path.isdir(train_root)):
        print(f"[CDD-11] Skipped invalid path: {root}")
        return

    clear_index = {
        os.path.basename(p): os.path.relpath(p, base)
        for p in glob.glob(os.path.join(clear_dir, "*"))
    }

    samples: List[Tuple[str, str, str]] = []
    for folder in os.listdir(train_root):
        if folder == "clear":
            continue
        fdir = os.path.join(train_root, folder)
        if not os.path.isdir(fdir):
            continue
        for p in glob.glob(os.path.join(fdir, "*")):
            key = os.path.basename(p)
            if key in clear_index:
                rel_deg   = os.path.relpath(p, base)
                rel_clear = clear_index[key]
                samples.append((rel_deg, rel_clear, folder))

    random.Random(seed).shuffle(samples)
    n_val = int(len(samples) * max(0.0, min(1.0, val_ratio)))
    val_samples   = samples[:n_val]
    train_samples = samples[n_val:]

    def _write(fn, arr):
        with open(fn, "w") as f:
            for rel_deg, rel_clear, folder in arr:
                f.write(f"{rel_deg} {rel_clear} {folder}\n")

    _write(os.path.join(out_dir, "cdd11_train.txt"), train_samples)
    _write(os.path.join(out_dir, "cdd11_val.txt"), val_samples)
    print(f"[CDD-11 split] train={len(train_samples)} val={len(val_samples)} -> {out_dir}")


def get_dataset_class_counts(dataset: Dataset) -> Dict[int, int]:
    """
    Helper to get class ID counts from a dataset with a get_all_labels method.
    """
    if hasattr(dataset, "get_all_labels"):
        labels = dataset.get_all_labels()
        if labels is not None and len(labels) > 0:
            unique, counts = np.unique(labels, return_counts=True)
            return dict(zip(unique.astype(int), counts.astype(int)))
    return {}


# =========================
# Paired Transform
# =========================
class PairedTransform:
    """
    Joint transforms for paired (degraded, clear) images.
    """
    def __init__(self, image_size=(512, 512), augment=False, normalize=True):
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize

        self.resize = T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC)
        self.to_tensor = T.ToTensor()
        self.norm_tf = T.Normalize(IMAGENET_MEAN, IMAGENET_STD) if normalize else (lambda x: x)

    def __call__(self, img_x: Image.Image, img_y: Optional[Image.Image]):
        x = self.resize(img_x)
        y = self.resize(img_y) if img_y else None

        if self.augment:
            x, y = _maybe_joint_flip(x, y)

        x = self.to_tensor(x)
        y_tensor = self.to_tensor(y) if y else None

        x = self.norm_tf(x)
        if y_tensor is not None:
            y_tensor = self.norm_tf(y_tensor)

        return x, y_tensor


# =========================
# CDD-11 Dataset
# =========================
class CDD11Dataset(Dataset):
    """
    CDD-11 paired degradation/clear dataset.

    Supports:
      - train/val via splits/cdd11_{train,val}.txt (auto-generated by make_cdd11_split)
      - test via:
          A) splits/cdd11_test.txt   OR
          B) auto-build from folder structure:
             <root>/test/clear and <root>/test/<style> paired by filename
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        list_dir: Optional[str] = None,
        image_size=(512, 512),
        transform: Optional[PairedTransform] = None,
        normalize: bool = True,
        max_samples_per_class: Optional[int] = None,
    ):
        super().__init__()
        base = os.path.abspath(root)

        self.base = base
        self.split = split

        if transform is None:
            transform = PairedTransform(image_size, augment=(split == "train"), normalize=normalize)
        self.transform = transform

        if list_dir is None:
            list_dir = os.path.join(base, "splits")
        os.makedirs(list_dir, exist_ok=True)

        list_file = os.path.join(list_dir, f"cdd11_{split}.txt")
        if split == "test" and not os.path.isfile(list_file):
            alt = os.path.join(base, f"cdd11_{split}.txt")
            if os.path.isfile(alt):
                list_file = alt

        self.samples: List[Tuple[str, str, str]] = []
        self.wids: List[int] = []

        # -------------------------
        # Load from list file
        # -------------------------
        if os.path.isfile(list_file):
            with open(list_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 3:
                        continue
                    rel_deg, rel_clear, folder = parts

                    style_label = folder if folder in STYLE2ID else "clear"
                    wid = STYLE2ID[style_label]

                    self.samples.append((os.path.join(base, rel_deg), os.path.join(base, rel_clear), folder))
                    self.wids.append(wid)

        # -------------------------
        # Auto-build test if needed
        # -------------------------
        elif split == "test":
            test_root = os.path.join(base, "test")
            clear_dir = os.path.join(test_root, "clear")

            if os.path.isdir(test_root) and os.path.isdir(clear_dir):
                clear_index = {
                    os.path.basename(p): os.path.relpath(p, base)
                    for p in glob.glob(os.path.join(clear_dir, "*"))
                }

                for folder in os.listdir(test_root):
                    if folder == "clear":
                        continue
                    fdir = os.path.join(test_root, folder)
                    if not os.path.isdir(fdir):
                        continue

                    for p in glob.glob(os.path.join(fdir, "*")):
                        key = os.path.basename(p)
                        if key not in clear_index:
                            continue

                        rel_deg = os.path.relpath(p, base)
                        rel_clear = clear_index[key]

                        style_label = folder if folder in STYLE2ID else "clear"
                        wid = STYLE2ID[style_label]

                        self.samples.append((os.path.join(base, rel_deg), os.path.join(base, rel_clear), folder))
                        self.wids.append(wid)

            if not self.samples and base != "DUMMY_PATH":
                print(
                    f"[CDD11Dataset] WARNING: No test samples found.\n"
                    f"Expected either {list_file} OR structure:\n"
                    f"  {base}/test/clear and {base}/test/<style>/... paired by filename"
                )

        else:
            if base != "DUMMY_PATH":
                print(f"[CDD11Dataset] WARNING: Missing split list {list_file} for split={split}. Dataset empty.")

        if max_samples_per_class is not None and self.samples:
            self._balance_dataset(max_samples_per_class)

    def _balance_dataset(self, max_limit: int):
        class_groups: Dict[int, List[int]] = {}
        for i, class_idx in enumerate(self.wids):
            class_groups.setdefault(class_idx, []).append(i)

        new_indices = []
        r = random.Random(42)
        for _, indices in class_groups.items():
            if len(indices) > max_limit:
                indices = r.sample(indices, max_limit)
            new_indices.extend(indices)

        new_indices.sort()
        self.samples = [self.samples[i] for i in new_indices]
        self.wids    = [self.wids[i] for i in new_indices]

    def get_all_labels(self) -> np.ndarray:
        return np.array(self.wids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p_deg, p_clear, folder = self.samples[i]

        img_x = Image.open(p_deg).convert("RGB")
        img_y = Image.open(p_clear).convert("RGB")

        x, y = self.transform(img_x, img_y)

        wid = self.wids[i]
        weather = CDD11_FOLDER2WEATHER.get(folder, "clear")
        sev = estimate_severity(x, weather)

        return {
            "input": x,
            "target": y,
            "style_exemplar": y,
            "wid": torch.tensor(wid, dtype=torch.long),
            "severity": torch.tensor([sev], dtype=torch.float32),
            "target_text": "clear_day",
            "source_weather": folder,
            "file": p_deg,
            "dataset": "cdd11",
        }


# =========================
# Benchmark size helper
# =========================
def _get_benchmark_size(cdd_root: str, split: str, normalize: bool) -> Optional[int]:
    if not (cdd_root and os.path.isdir(cdd_root)):
        print(f"Warning: CDD root invalid for split '{split}'. Skipping benchmark.")
        return None

    try:
        temp_cdd = CDD11Dataset(cdd_root, split=split, normalize=normalize, max_samples_per_class=None)
        counts = get_dataset_class_counts(temp_cdd)
        valid_counts = list(counts.values())

        if valid_counts:
            benchmark_size = min(valid_counts)
            print(f"Determined downsampling benchmark size for CDD11 '{split}': {benchmark_size} samples/class.")
            return benchmark_size

        print(f"Warning: Could not determine benchmark size for split '{split}'. Disabling downsampling.")
        return None
    except Exception as e:
        print(f"Error calculating benchmark size for '{split}': {e}")
        return None


# =========================
# CDD-11 loader 
# =========================
def load_combined_dataset(
    cdd_train_root: Optional[str],
    cdd_test_root: Optional[str] = None,
    cdd_val_ratio: float = 0.1,
    split: str = "train",
    use_ref: bool = True,                   # kept for compatibility (ignored)
    image_size=(512, 512),
    batch_size: int = 4,
    workers: int = 4,
    distributed: bool = False,
    normalize: bool = True,
    train_transform: Optional[PairedTransform] = None,
    val_transform: Optional[PairedTransform] = None
):
    """
    CDD-11 ONLY.
    Preserves the original function signature so your existing scripts don't break.
    """

    valid_cdd_train = bool(cdd_train_root) and (cdd_train_root != "DUMMY_PATH") and os.path.isdir(cdd_train_root)
    valid_cdd_test  = bool(cdd_test_root)  and (cdd_test_root  != "DUMMY_PATH") and os.path.isdir(cdd_test_root)

    # If split=test but user only passed a single cdd root, fall back to it
    if split == "test" and (not valid_cdd_test) and valid_cdd_train:
        cdd_test_root = cdd_train_root
        valid_cdd_test = True

    if train_transform is None:
        train_transform = PairedTransform(image_size, augment=True, normalize=normalize)
    if val_transform is None:
        val_transform = PairedTransform(image_size, augment=False, normalize=normalize)

    train_sets: List[Dataset] = []
    val_sets: List[Dataset] = []
    test_sets: List[Dataset] = []

    eval_info = {
        "cdd11_val": None,
        "cdd11_test": None
    }

    # Prepare CDD splits for train/val
    list_dir = None
    if valid_cdd_train:
        list_dir = os.path.join(os.path.abspath(cdd_train_root), "splits")
        os.makedirs(list_dir, exist_ok=True)

        train_list = os.path.join(list_dir, "cdd11_train.txt")
        val_list   = os.path.join(list_dir, "cdd11_val.txt")

        if split in ["train", "val"]:
            if not (os.path.isfile(train_list) and os.path.isfile(val_list)):
                make_cdd11_split(cdd_train_root, out_dir=list_dir, val_ratio=cdd_val_ratio)

    # Benchmark sizes (optional)
    BENCHMARK_SIZE_TRAIN_VAL = None
    BENCHMARK_SIZE_TEST = None

    if valid_cdd_train and split in ["train", "val"]:
        BENCHMARK_SIZE_TRAIN_VAL = _get_benchmark_size(cdd_train_root, "train", normalize)

    if valid_cdd_test and split == "test":
        BENCHMARK_SIZE_TEST = _get_benchmark_size(cdd_test_root, "test", normalize)

    # Build datasets
    if valid_cdd_train and split in ["train", "val"]:
        c_train = CDD11Dataset(
            cdd_train_root,
            split="train",
            list_dir=list_dir,
            image_size=image_size,
            transform=train_transform,
            normalize=normalize,
            max_samples_per_class=BENCHMARK_SIZE_TRAIN_VAL
        )
        c_val = CDD11Dataset(
            cdd_train_root,
            split="val",
            list_dir=list_dir,
            image_size=image_size,
            transform=val_transform,
            normalize=normalize,
            max_samples_per_class=BENCHMARK_SIZE_TRAIN_VAL
        )

        if split == "train" and len(c_train) > 0:
            train_sets.append(c_train)

        if len(c_val) > 0:
            val_sets.append(c_val)
            eval_info["cdd11_val"] = c_val

    if valid_cdd_test and split == "test":
        test_list_dir = os.path.join(os.path.abspath(cdd_test_root), "splits")
        os.makedirs(test_list_dir, exist_ok=True)

        c_test = CDD11Dataset(
            cdd_test_root,
            split="test",
            list_dir=test_list_dir,
            image_size=image_size,
            transform=val_transform,
            normalize=normalize,
            max_samples_per_class=BENCHMARK_SIZE_TEST
        )

        if len(c_test) > 0:
            test_sets.append(c_test)
            eval_info["cdd11_test"] = c_test

    assert train_sets or val_sets or test_sets, (
        "No datasets enabled. Check roots and folder structure.\n"
        f"split={split}\n"
        f"cdd_train_root={cdd_train_root} (valid={valid_cdd_train})\n"
        f"cdd_test_root={cdd_test_root} (valid={valid_cdd_test})\n"
        "Expected:\n"
        "  <root>/train/clear and <root>/train/<style>\n"
        "  plus splits/cdd11_train.txt & splits/cdd11_val.txt (auto-generated)\n"
        "Test expects either splits/cdd11_test.txt OR <root>/test/clear + <root>/test/<style>"
    )

    # DataLoaders
    train_loader = None
    if train_sets:
        train_ds = train_sets[0]
        train_sampler = _maybe_build_sampler(train_ds, distributed, shuffle=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False
        )

    val_loader = None
    if val_sets:
        val_ds = val_sets[0]
        val_sampler = _maybe_build_sampler(val_ds, distributed, shuffle=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=max(1, batch_size // 2),
            sampler=val_sampler,
            num_workers=max(1, workers // 2),
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False
        )

    test_loader = None
    if test_sets:
        test_ds = test_sets[0]
        test_sampler = _maybe_build_sampler(test_ds, distributed, shuffle=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=max(1, batch_size // 2),
            sampler=test_sampler,
            num_workers=max(1, workers // 2),
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False
        )

    return train_loader, val_loader, test_loader, eval_info