from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from road_extraction.data import MassachusettsRoadsDataset
from road_extraction.pipelines import load_experiment_model, predict_single_image
from road_extraction.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--data-root", default="data/massachusetts")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output-dir", default="experiments/compare")
    args = parser.parse_args()

    baseline_dir = (project_root / args.baseline).resolve()
    candidate_dir = (project_root / args.candidate).resolve()
    data_root = (project_root / args.data_root).resolve()
    output_dir = ensure_dir(project_root / args.output_dir)

    baseline_model, baseline_cfg, _ = load_experiment_model(baseline_dir)
    candidate_model, candidate_cfg, _ = load_experiment_model(candidate_dir)
    dataset = MassachusettsRoadsDataset(root=data_root, split="test", include_masks=True)
    picks = random.sample(range(len(dataset.records)), min(args.samples, len(dataset.records)))

    for index in picks:
        sample = dataset.get_full_sample(index)
        baseline = predict_single_image(
            baseline_model,
            sample["image_path"],
            patch_size=baseline_cfg["dataset"].get("patch_size", 512),
            target_size=baseline_cfg["dataset"].get("target_size", 1536),
        )
        candidate = predict_single_image(
            candidate_model,
            sample["image_path"],
            patch_size=candidate_cfg["dataset"].get("patch_size", 512),
            target_size=candidate_cfg["dataset"].get("target_size", 1536),
        )
        mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        gain = (candidate["prediction"] == 1) & (baseline["prediction"] == 0) & (mask == 1)
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        axes[0, 0].imshow(baseline["image"])
        axes[0, 0].set_title("Image")
        axes[0, 1].imshow(mask, cmap="gray")
        axes[0, 1].set_title("Mask")
        axes[0, 2].imshow(baseline["prediction"], cmap="gray")
        axes[0, 2].set_title("Baseline")
        axes[0, 3].imshow(candidate["prediction"], cmap="gray")
        axes[0, 3].set_title("Candidate")
        axes[1, 0].imshow(baseline["probability"], cmap="viridis")
        axes[1, 0].set_title("Baseline Prob")
        axes[1, 1].imshow(candidate["probability"], cmap="viridis")
        axes[1, 1].set_title("Candidate Prob")
        gain_img = np.zeros((*gain.shape, 3), dtype=np.float32)
        gain_img[gain] = [1, 1, 0]
        axes[1, 2].imshow(gain_img)
        axes[1, 2].set_title("Recall Gain")
        axes[1, 3].text(0.1, 0.5, f"gain_pixels={int(gain.sum())}", fontsize=12)
        axes[1, 3].axis("off")
        for axis in axes.flat:
            if axis is not axes[1, 3]:
                axis.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"comparison_{Path(sample['image_path']).stem}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
