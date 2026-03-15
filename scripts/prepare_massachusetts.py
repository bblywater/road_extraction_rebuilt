from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from road_extraction.data.datasets import find_mask_path, pad_to_size


def copy_tree(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source_dir.iterdir()):
        if source_path.is_file():
            target_path = target_dir / source_path.name
            if not target_path.exists():
                shutil.copy2(source_path, target_path)


def write_patch_dataset(dataset_root: Path, patch_root: Path, splits: list[str], patch_size: int, target_size: int) -> None:
    grid = target_size // patch_size
    for split in splits:
        image_dir = dataset_root / split / "data"
        mask_dir = dataset_root / split / "label"
        patch_image_dir = patch_root / split / "data"
        patch_mask_dir = patch_root / split / "label"
        patch_image_dir.mkdir(parents=True, exist_ok=True)
        patch_mask_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            list(image_dir.glob("*.tif"))
            + list(image_dir.glob("*.tiff"))
            + list(image_dir.glob("*.png"))
            + list(image_dir.glob("*.jpg"))
        )
        written = 0
        for image_path in image_files:
            mask_path = find_mask_path(image_path, mask_dir)
            if mask_path is None:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image, _ = pad_to_size(image, target_size, is_mask=False)
            mask, _ = pad_to_size(mask, target_size, is_mask=True)
            stem = image_path.stem
            for row in range(grid):
                for col in range(grid):
                    y0 = row * patch_size
                    x0 = col * patch_size
                    image_patch = image[y0 : y0 + patch_size, x0 : x0 + patch_size]
                    mask_patch = mask[y0 : y0 + patch_size, x0 : x0 + patch_size]
                    patch_name = f"{stem}_r{row}_c{col}.png"
                    cv2.imwrite(str(patch_image_dir / patch_name), cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(patch_mask_dir / patch_name), mask_patch)
                    written += 1
        print({"split": split, "patches_written": written, "patch_root": str(patch_root / split)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="../../Massachusetts Roads Dataset/tiff")
    parser.add_argument("--target", default="data/massachusetts")
    parser.add_argument("--patch-target", default="data/massachusetts_patches")
    parser.add_argument("--generate-patches", action="store_true")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--target-size", type=int, default=1536)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    source_root = Path(args.source)
    if not source_root.is_absolute():
        source_root = (project_root / source_root).resolve()
    target_root = Path(args.target)
    if not target_root.is_absolute():
        target_root = (project_root / target_root).resolve()
    patch_root = Path(args.patch_target)
    if not patch_root.is_absolute():
        patch_root = (project_root / patch_root).resolve()

    mapping = {
        "train": ("train", "train_labels"),
        "val": ("val", "val_labels"),
        "test": ("test", "test_labels"),
    }
    for split, (image_name, label_name) in mapping.items():
        if split not in args.splits:
            continue
        copy_tree(source_root / image_name, target_root / split / "data")
        copy_tree(source_root / label_name, target_root / split / "label")

    if args.generate_patches:
        write_patch_dataset(
            dataset_root=target_root,
            patch_root=patch_root,
            splits=args.splits,
            patch_size=args.patch_size,
            target_size=args.target_size,
        )

    print(
        {
            "source": str(source_root),
            "target": str(target_root),
            "patch_target": str(patch_root),
            "generate_patches": bool(args.generate_patches),
        }
    )


if __name__ == "__main__":
    main()
