from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
THESIS_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = THESIS_ROOT / "figures" / "generated"
DATA_ROOT = ROOT / "data" / "massachusetts"
EXPERIMENTS_ROOT = ROOT / "experiments"


MODEL_INFO = [
    ("vanilla_unet_massachusetts_full", "Vanilla U-Net"),
    ("dilated_baseline_unet_massachusetts_full", "Dilated Baseline"),
    ("baseline_unet_massachusetts_full", "Baseline U-Net"),
    ("unetpp_massachusetts_full", "UNet++"),
    ("dlgu_net_improved_full", "DLGU-Net"),
    ("residual_vanilla_unet_massachusetts_full", "Residual Vanilla"),
    ("attention_unet_massachusetts_full", "Attention U-Net"),
    ("resnet34_unet_massachusetts_full", "ResNet34 U-Net"),
    ("ddu_net_full_stable", "DDU-Net"),
]

CORE_HISTORY_MODELS = [
    ("baseline_unet_massachusetts_full", "Baseline U-Net"),
    ("dlgu_net_improved_full", "DLGU-Net"),
    ("dilated_baseline_unet_massachusetts_full", "Dilated Baseline"),
    ("vanilla_unet_massachusetts_full", "Vanilla U-Net"),
]

QUALITATIVE_MODELS = [
    ("baseline_unet_massachusetts_full", "Baseline"),
    ("dlgu_net_improved_full", "DLGU"),
    ("dilated_baseline_unet_massachusetts_full", "Dilated"),
    ("vanilla_unet_massachusetts_full", "Vanilla"),
]


def ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L"))
    return (mask > 127).astype(np.uint8)


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=240, bbox_inches="tight")
    plt.close(fig)


def load_summary(exp_dir: str, split: str = "full_evaluation") -> dict:
    path = EXPERIMENTS_ROOT / exp_dir / split / "summary_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_history(exp_dir: str) -> dict:
    path = EXPERIMENTS_ROOT / exp_dir / "training_history.json"
    return json.loads(path.read_text(encoding="utf-8"))


def select_density_examples() -> list[tuple[str, float]]:
    label_dir = DATA_ROOT / "test" / "label"
    items = []
    for label_path in sorted(label_dir.glob("*.tif")):
        ratio = float(load_mask(label_path).mean())
        items.append((label_path.stem, ratio))
    items.sort(key=lambda item: item[1])
    if len(items) < 3:
        raise RuntimeError("Not enough test images to generate dataset examples.")
    indices = [max(0, len(items) // 8), len(items) // 2, min(len(items) - 1, len(items) * 7 // 8)]
    return [items[i] for i in indices]


def create_dataset_examples() -> tuple[str, str]:
    examples = select_density_examples()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    density_names = ["Sparse roads", "Medium roads", "Dense roads"]
    for col, ((stem, ratio), density_name) in enumerate(zip(examples, density_names)):
        rgb = load_rgb(DATA_ROOT / "test" / "data" / f"{stem}.tiff")
        mask = load_mask(DATA_ROOT / "test" / "label" / f"{stem}.tif")
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"{density_name}\n{stem}")
        axes[0, col].axis("off")
        axes[1, col].imshow(mask, cmap="gray")
        axes[1, col].set_title(f"Road ratio = {ratio:.3f}")
        axes[1, col].axis("off")
    save_fig(fig, "dataset_examples.png")
    return examples[1][0], f"{examples[0][0]}, {examples[1][0]}, {examples[2][0]}"


def add_grid(image: np.ndarray, step: int = 512) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    color = np.array([255, 80, 80], dtype=np.uint8)
    for x in range(step, w, step):
        canvas[:, max(0, x - 2):min(w, x + 2)] = color
    for y in range(step, h, step):
        canvas[max(0, y - 2):min(h, y + 2), :] = color
    return canvas


def build_patch_montage(image: np.ndarray, patch_size: int = 512, gap: int = 18) -> np.ndarray:
    canvas_h = patch_size * 3 + gap * 2
    canvas_w = patch_size * 3 + gap * 2
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    border_color = np.array([220, 70, 70], dtype=np.uint8)
    for row in range(3):
        for col in range(3):
            y1 = row * patch_size
            x1 = col * patch_size
            patch = image[y1:y1 + patch_size, x1:x1 + patch_size]
            dst_y = row * (patch_size + gap)
            dst_x = col * (patch_size + gap)
            canvas[dst_y:dst_y + patch_size, dst_x:dst_x + patch_size] = patch
            canvas[dst_y:dst_y + 6, dst_x:dst_x + patch_size] = border_color
            canvas[dst_y + patch_size - 6:dst_y + patch_size, dst_x:dst_x + patch_size] = border_color
            canvas[dst_y:dst_y + patch_size, dst_x:dst_x + 6] = border_color
            canvas[dst_y:dst_y + patch_size, dst_x + patch_size - 6:dst_x + patch_size] = border_color
    return canvas


def create_preprocess_pipeline(stem: str) -> None:
    rgb = load_rgb(DATA_ROOT / "test" / "data" / f"{stem}.tiff")
    mask = load_mask(DATA_ROOT / "test" / "label" / f"{stem}.tif")
    pad_h = 1536 - rgb.shape[0]
    pad_w = 1536 - rgb.shape[1]
    rgb_pad = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    mask_pad = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="reflect")
    grid_rgb = add_grid(rgb_pad)
    patch_montage = build_patch_montage(rgb_pad)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original image")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("Binary road mask")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(grid_rgb)
    axes[1, 0].set_title("Reflected padding to 1536x1536 and 3x3 grid")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(patch_montage)
    axes[1, 1].set_title("Nine 512x512 patches used for training")
    axes[1, 1].axis("off")
    save_fig(fig, "preprocessing_pipeline.png")

    fig2, ax2 = plt.subplots(figsize=(5.4, 5.4))
    ax2.imshow(add_grid(np.dstack([mask_pad * 255] * 3)))
    ax2.set_title("Patch grid on padded road mask")
    ax2.axis("off")
    save_fig(fig2, "patch_grid_mask.png")


def create_engineering_speedup() -> None:
    labels = ["Online crop", "Offline patches", "Logging tuned"]
    values = [409.3, 119.1, 113.5]
    colors = ["#9d9d9d", "#5c88da", "#2f6f3e"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Seconds per epoch")
    ax.set_title("Engineering optimization on baseline training")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 6, f"{value:.1f}", ha="center", va="bottom")
    save_fig(fig, "engineering_speedup.png")


def collect_model_results() -> list[dict]:
    records = []
    for exp_dir, display_name in MODEL_INFO:
        summary = load_summary(exp_dir)
        val_summary = load_summary(exp_dir, split="full_evaluation_val")
        avg_metrics = summary["average_metrics"]
        val_avg_metrics = val_summary["average_metrics"]
        params = summary["model_info"]["parameters"]
        records.append(
            {
                "dir": exp_dir,
                "name": display_name,
                "params_m": params / 1_000_000.0,
                "val_iou": float(val_avg_metrics["iou"]),
                "test_iou": float(avg_metrics["iou"]),
                "test_f1": float(avg_metrics["f1"]),
            }
        )
    return records


def create_model_performance_figures() -> None:
    records = collect_model_results()
    records_sorted = sorted(records, key=lambda item: item["test_iou"], reverse=True)
    names = [record["name"] for record in records_sorted]
    ious = [record["test_iou"] for record in records_sorted]
    f1s = [record["test_f1"] for record in records_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    axes[0].barh(names, ious, color="#4c72b0")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Test IoU")
    axes[0].set_title("Test IoU comparison")
    axes[0].grid(axis="x", linestyle="--", alpha=0.3)
    for y, value in enumerate(ious):
        axes[0].text(value + 0.0005, y, f"{value:.4f}", va="center", fontsize=8)

    axes[1].barh(names, f1s, color="#dd8452")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Test F1")
    axes[1].set_title("Test F1 comparison")
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)
    for y, value in enumerate(f1s):
        axes[1].text(value + 0.0005, y, f"{value:.4f}", va="center", fontsize=8)
    save_fig(fig, "model_performance.png")

    fig2, ax2 = plt.subplots(figsize=(7.4, 5.4))
    for record in records:
        ax2.scatter(record["params_m"], record["test_iou"], s=70, alpha=0.85)
        ax2.text(record["params_m"] + 0.18, record["test_iou"] + 0.00015, record["name"], fontsize=8)
    ax2.set_xlabel("Parameters (M)")
    ax2.set_ylabel("Test IoU")
    ax2.set_title("Accuracy-parameter trade-off")
    ax2.grid(linestyle="--", alpha=0.3)
    save_fig(fig2, "model_tradeoff.png")


def create_training_curve_comparison() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for exp_dir, display_name in CORE_HISTORY_MODELS:
        history = load_history(exp_dir)
        epochs = np.arange(1, len(history["val_iou"]) + 1)
        axes[0].plot(epochs, history["val_iou"], marker="o", linewidth=1.8, label=display_name)
        axes[1].plot(epochs, history["train_loss"], marker="o", linewidth=1.8, label=display_name)
    axes[0].set_title("Validation IoU across epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val IoU")
    axes[0].grid(linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Training loss across epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train loss")
    axes[1].grid(linestyle="--", alpha=0.3)
    axes[1].legend(fontsize=8)
    save_fig(fig, "training_curves_comparison.png")


def crop_panel(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    return image.crop(box)


def create_qualitative_comparison() -> None:
    samples = [1, 4]
    boxes = {
        "image": (16, 49, 665, 697),
        "mask": (737, 49, 1386, 697),
        "pred": (16, 757, 665, 1405),
    }

    fig, axes = plt.subplots(len(samples), 2 + len(QUALITATIVE_MODELS), figsize=(15.2, 4.9 * len(samples)))
    if len(samples) == 1:
        axes = np.array([axes])

    titles = ["Image", "Mask"] + [display_name for _, display_name in QUALITATIVE_MODELS]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title)

    for row, sample_id in enumerate(samples):
        baseline_img = Image.open(
            EXPERIMENTS_ROOT
            / "baseline_unet_massachusetts_full"
            / "full_evaluation"
            / "sample_visualizations"
            / f"sample_{sample_id}.png"
        ).convert("RGB")
        axes[row, 0].imshow(crop_panel(baseline_img, boxes["image"]))
        axes[row, 0].axis("off")
        axes[row, 1].imshow(crop_panel(baseline_img, boxes["mask"]), cmap="gray")
        axes[row, 1].axis("off")
        for col, (exp_dir, _) in enumerate(QUALITATIVE_MODELS, start=2):
            vis_img = Image.open(
                EXPERIMENTS_ROOT / exp_dir / "full_evaluation" / "sample_visualizations" / f"sample_{sample_id}.png"
            ).convert("RGB")
            axes[row, col].imshow(crop_panel(vis_img, boxes["pred"]))
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"Sample {sample_id}", rotation=90, fontsize=10)
    save_fig(fig, "qualitative_comparison.png")


def main() -> None:
    ensure_dir()
    medium_stem, _ = create_dataset_examples()
    create_preprocess_pipeline(medium_stem)
    create_engineering_speedup()
    create_model_performance_figures()
    create_training_curve_comparison()
    create_qualitative_comparison()
    print(f"Generated figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
