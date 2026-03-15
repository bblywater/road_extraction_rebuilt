from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from road_extraction.pipelines import load_experiment_model, predict_single_image
from road_extraction.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment)
    if not experiment_dir.is_absolute():
        experiment_dir = (project_root / experiment_dir).resolve()
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = (project_root / image_path).resolve()
    output_dir = ensure_dir(args.output_dir or experiment_dir / "single_image")

    model, config, _ = load_experiment_model(experiment_dir)
    result = predict_single_image(
        model,
        image_path,
        patch_size=config["dataset"].get("patch_size", 512),
        target_size=config["dataset"].get("target_size", 1536),
    )

    mask_path = output_dir / f"{image_path.stem}_road_mask.png"
    Image.fromarray((result["prediction"] * 255).astype("uint8")).save(mask_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(result["image"])
    axes[0].set_title("Image")
    axes[0].axis("off")
    axes[1].imshow(result["prediction"], cmap="gray")
    axes[1].set_title(f"Prediction coverage={result['coverage']:.4f}")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{image_path.stem}_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(
        {
            "mask_path": str(mask_path),
            "coverage": result["coverage"],
            "road_pixels": result["road_pixels"],
            "shape": result["original_shape"],
        }
    )


if __name__ == "__main__":
    main()
