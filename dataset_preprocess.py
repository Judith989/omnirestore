#!/usr/bin/env python3
# dataset_preprocess.py
# -----------------------------------------------------------------------------
# Prepares ACDC (rgb_anon) and CDD-11 into a unified "prepared" directory:
#   prepared/cdd11/{train,val,test}/...
#   prepared/acdc/{train,val,test}/...
# - CDD-11_train is stratified-split into train/val by --split_val.
# - CDD-11_test is copied to test (no split).
# - ACDC rgb_anon is mirrored as-is (no split), preserving condition/split/sequence structure.
# - Images are resized to --resize with bicubic.
# - Optional --normalize is recorded into a meta JSON (we do NOT bake normalization into images).
# -----------------------------------------------------------------------------

from __future__ import annotations
import argparse, os, sys, json, random, shutil
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None  # allow large files

# Canonical map (CDD-11 folders -> canonical weather)
CDD11_FOLDER2WEATHER = {
    "clear": "clear",
    "snow": "snow",
    "rain": "rain",
    "haze": "haze",
    "low":  "low",
    "low_haze": "low",
    "low_rain": "low",
    "low_snow": "low",
    "haze_rain": "rain",
    "haze_snow": "snow",
    "low_haze_rain": "low",
    "low_haze_snow": "low",
}

# Basic image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_resized(src: Path, dst: Path, size: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im = im.resize((size, size), Image.BICUBIC)
        im.save(dst)


def copy_tree_resized(src_root: Path, dst_root: Path, resize: int):
    """
    Recursively copy images from src_root to dst_root, resizing images.
    Non-image files are skipped.
    """
    files = [p for p in src_root.rglob("*") if p.is_file() and is_image(p)]
    for p in tqdm(files, desc=f"Copy {src_root.name} -> {dst_root.name}"):
        rel = p.relative_to(src_root)
        dst = dst_root / rel
        save_resized(p, dst, resize)


def index_cdd11_pairs(train_root: Path) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    For CDD-11_train root that contains 'clear' + weather folders:
      returns dict: weather_folder -> list of (degraded_path, clear_path)
    """
    clear_dir = train_root / "clear"
    if not clear_dir.is_dir():
        raise FileNotFoundError(f"CDD-11_train must contain a 'clear' folder: {clear_dir}")

    clear_map = {p.name: p for p in clear_dir.iterdir() if p.is_file() and is_image(p)}
    pairs: Dict[str, List[Tuple[Path, Path]]] = {}

    for folder in train_root.iterdir():
        if not folder.is_dir() or folder.name == "clear":
            continue
        f_name = folder.name
        pairs.setdefault(f_name, [])
        for p in folder.iterdir():
            if not p.is_file() or not is_image(p):
                continue
            if p.name in clear_map:
                pairs[f_name].append((p, clear_map[p.name]))

    # remove empties
    pairs = {k: v for k, v in pairs.items() if v}
    if not pairs:
        raise RuntimeError(f"No CDD-11 pairs found under {train_root}")

    return pairs


def stratified_split(pairs_per_folder: Dict[str, List[Tuple[Path, Path]]],
                     val_ratio: float,
                     seed: int = 42) -> Tuple[
                         Dict[str, List[Tuple[Path, Path]]],
                         Dict[str, List[Tuple[Path, Path]]]
                     ]:
    """
    Split pairs per weather folder into train/val by val_ratio.
    """
    rng = random.Random(seed)
    train_split, val_split = {}, {}
    for folder, lst in pairs_per_folder.items():
        n = len(lst)
        idxs = list(range(n))
        rng.shuffle(idxs)
        n_val = int(round(n * val_ratio))
        val_idx = set(idxs[:n_val])
        tr, va = [], []
        for i, pair in enumerate(lst):
            (va if i in val_idx else tr).append(pair)
        if tr: train_split[folder] = tr
        if va: val_split[folder] = va
    return train_split, val_split


def write_cdd11_split(split_pairs: Dict[str, List[Tuple[Path, Path]]],
                      dst_split_root: Path,
                      resize: int):
    """
    Write a CDD-11 split to dst_split_root (has subfolders per weather + 'clear').
    Keep the same filename for degraded and its matching clear counterpart.
    """
    # Ensure 'clear' exists
    safe_mkdir(dst_split_root / "clear")

    total = sum(len(v) for v in split_pairs.values())
    pbar = tqdm(total=total, desc=f"Write CDD-11 {dst_split_root.name}")

    for folder, pairs in split_pairs.items():
        weather_dir = dst_split_root / folder
        safe_mkdir(weather_dir)
        for p_deg, p_clr in pairs:
            # degraded
            dst_deg = weather_dir / p_deg.name
            save_resized(p_deg, dst_deg, resize)
            # clear
            dst_clr = (dst_split_root / "clear" / p_clr.name)
            if not dst_clr.exists():  # only copy once if multiple degraded share the same clear
                save_resized(p_clr, dst_clr, resize)
            pbar.update(1)

    pbar.close()


def process_cdd11(cdd11_train: Path, cdd11_test: Path, out_root: Path, split_val: float, resize: int, seed: int):
    dst_cdd = out_root / "cdd11"
    (dst_cdd / "train").mkdir(parents=True, exist_ok=True)
    (dst_cdd / "val").mkdir(parents=True, exist_ok=True)
    (dst_cdd / "test").mkdir(parents=True, exist_ok=True)

    # --- Train/Val from CDD-11_train ---
    pairs = index_cdd11_pairs(cdd11_train)
    tr_pairs, va_pairs = stratified_split(pairs, val_ratio=split_val, seed=seed)
    write_cdd11_split(tr_pairs, dst_cdd / "train", resize)
    write_cdd11_split(va_pairs, dst_cdd / "val", resize)

    # --- Test from CDD-11_test (copy/no split) ---
    # Keep folders as-is (including 'clear' if present)
    for folder in [p for p in cdd11_test.iterdir() if p.is_dir()]:
        copy_tree_resized(folder, (dst_cdd / "test" / folder.name), resize)


def list_acdc_splits(rgb_root: Path) -> List[str]:
    # detect available splits under any condition (train, val, test)
    splits = set()
    for cond in ["fog", "night", "rain", "snow"]:
        cdir = rgb_root / cond
        if not cdir.is_dir(): 
            continue
        for sp in cdir.iterdir():
            if sp.is_dir():
                splits.add(sp.name)
    # stable order
    return [s for s in ["train", "val", "test"] if s in splits] + \
           [s for s in sorted(splits) if s not in {"train", "val", "test"}]


def process_acdc(rgb_root: Path, out_root: Path, resize: int):
    """
    Mirror rgb_anon structure:
      rgb_anon/{cond}/{split}/{sequence}/... -> prepared/acdc/{split}/{cond}/{sequence}/...
    """
    dst_acdc = out_root / "acdc"
    dst_acdc.mkdir(parents=True, exist_ok=True)

    splits = list_acdc_splits(rgb_root)
    if not splits:
        print(f"[warn] No ACDC splits found under {rgb_root}. Skipping ACDC processing.")
        return

    for split in splits:
        for cond in ["fog", "night", "rain", "snow"]:
            src_cond_split = rgb_root / cond / split
            if not src_cond_split.is_dir():
                continue
            # copy sequences recursively
            dst = dst_acdc / split / cond
            files = [p for p in src_cond_split.rglob("*") if p.is_file() and is_image(p)]
            for p in tqdm(files, desc=f"ACDC {cond}/{split} -> prepared/acdc/{split}/{cond}"):
                rel = p.relative_to(src_cond_split)
                save_resized(p, dst / rel, resize)


def write_meta(out_root: Path, args: argparse.Namespace):
    meta = {
        "resize": args.resize,
        "normalize_flag": args.normalize,
        "note": "Normalization is not baked into images. Use transforms.Normalize at training time.",
        "sources": {
            "cdd11_train": str(args.cdd11_train) if args.cdd11_train else None,
            "cdd11_test": str(args.cdd11_test) if args.cdd11_test else None,
            "acdc_rgb": str(args.acdc_rgb) if args.acdc_rgb else None,
        }
    }
    with open(out_root / "prepared_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[meta] wrote {out_root / 'prepared_meta.json'}")


def parse_args():
    ap = argparse.ArgumentParser(description="Preprocess ACDC (rgb_anon) and CDD-11 into a unified prepared directory.")
    ap.add_argument("--cdd11_train", type=str, required=True, help="Path to CDD-11_train folder")
    ap.add_argument("--cdd11_test",  type=str, required=True, help="Path to CDD-11_test folder")
    ap.add_argument("--acdc_rgb",    type=str, required=True, help="Path to ACDC 'rgb_anon' root")
    ap.add_argument("--output_dir",  type=str, required=True, help="Output root (e.g., ./data/prepared)")
    ap.add_argument("--split_val",   type=float, default=0.10, help="Validation ratio for CDD-11_train")
    ap.add_argument("--resize",      type=int, default=256, help="Resize (square) size for saved images")
    ap.add_argument("--normalize",   action="store_true", help="(Recorded only) Use Normalize() at train-time")
    ap.add_argument("--seed",        type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()

    cdd11_train = Path(args.cdd11_train).resolve()
    cdd11_test  = Path(args.cdd11_test).resolve()
    acdc_rgb    = Path(args.acdc_rgb).resolve()
    out_root    = Path(args.output_dir).resolve()

    for p, name in [(cdd11_train, "CDD-11_train"), (cdd11_test, "CDD-11_test"), (acdc_rgb, "rgb_anon")]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {name} at: {p}")

    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[prep] CDD-11 -> {out_root / 'cdd11'}  (split val={args.split_val})")
    process_cdd11(cdd11_train, cdd11_test, out_root, split_val=args.split_val, resize=args.resize, seed=args.seed)

    print(f"[prep] ACDC/rgb_anon -> {out_root / 'acdc'}")
    process_acdc(acdc_rgb, out_root, resize=args.resize)

    write_meta(out_root, args)
    print("[done] preprocessing completed successfully.")


if __name__ == "__main__":
    main()
