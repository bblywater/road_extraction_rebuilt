from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def summarize_split(split_dir: Path) -> dict:
    image_dir = split_dir / "data"
    label_dir = split_dir / "label"
    images = sorted(list(image_dir.glob("*.*"))) if image_dir.exists() else []
    labels = sorted(list(label_dir.glob("*.*"))) if label_dir.exists() else []
    payload = {"images": len(images), "labels": len(labels), "sample_shape": None, "mask_values": None}
    if images:
        sample = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        payload["sample_shape"] = tuple(sample.shape) if sample is not None else None
    if labels:
        mask = cv2.imread(str(labels[0]), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            payload["mask_values"] = np.unique(mask).tolist()[:10]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/massachusetts")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    root = Path(args.root)
    if not root.is_absolute():
        root = (project_root / root).resolve()
    summary = {split: summarize_split(root / split) for split in ("train", "val", "test")}
    print(summary)


if __name__ == "__main__":
    main()
