from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .io import ensure_dir


def plot_training_curves(history: dict[str, list[float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, history.get("train_loss", []), label="train")
    axes[0, 0].plot(epochs, history.get("val_loss", []), label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history.get("train_iou", []), label="train")
    axes[0, 1].plot(epochs, history.get("val_iou", []), label="val")
    axes[0, 1].set_title("IoU")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history.get("train_f1", []), label="train")
    axes[1, 0].plot(epochs, history.get("val_f1", []), label="val")
    axes[1, 0].set_title("F1")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    learning_rate = history.get("lr", history.get("learning_rate", []))
    axes[1, 1].plot(range(1, len(learning_rate) + 1), learning_rate)
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_evaluation_summary(
    iou_values: Iterable[float],
    precision_values: Iterable[float],
    recall_values: Iterable[float],
    metric_means: dict[str, float],
    confusion: np.ndarray,
    output_dir: str | Path,
) -> None:
    output_dir = ensure_dir(output_dir)
    iou_values = np.asarray(list(iou_values))
    precision_values = np.asarray(list(precision_values))
    recall_values = np.asarray(list(recall_values))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(iou_values, bins=20, color="#74c0fc", edgecolor="black", alpha=0.8)
    axes[0, 0].axvline(metric_means["iou"], color="red", linestyle="--", linewidth=2)
    axes[0, 0].set_title("IoU Distribution")

    axes[0, 1].scatter(precision_values, recall_values, alpha=0.6, color="#2b8a3e")
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Precision vs Recall")

    metric_names = ["iou", "f1", "precision", "recall", "accuracy", "dice"]
    metric_values = [metric_means[name] for name in metric_names]
    axes[1, 0].bar(metric_names, metric_values, color=["#74c0fc", "#9775fa", "#ffa94d", "#63e6be", "#51cf66", "#ffd43b"])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Metric Means")

    sns.heatmap(confusion, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=axes[1, 1])
    axes[1, 1].set_title("Confusion Matrix")
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_sample_visuals(samples: list[dict[str, np.ndarray]], output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    for index, sample in enumerate(samples, start=1):
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes[0, 0].imshow(sample["image"])
        axes[0, 0].set_title("Image")
        axes[0, 1].imshow(sample["mask"], cmap="gray")
        axes[0, 1].set_title("Mask")
        heatmap = axes[0, 2].imshow(sample["probability"], cmap="viridis", vmin=0, vmax=1)
        axes[0, 2].set_title("Probability")
        plt.colorbar(heatmap, ax=axes[0, 2], fraction=0.046, pad=0.04)
        axes[1, 0].imshow(sample["prediction"], cmap="gray")
        axes[1, 0].set_title("Prediction")
        axes[1, 1].imshow(sample["error"])
        axes[1, 1].set_title("Error")
        axes[1, 2].text(0.1, 0.5, f"IoU: {sample['iou']:.4f}\nF1: {sample['f1']:.4f}", fontsize=12)
        axes[1, 2].axis("off")
        for axis in axes.flat:
            if axis is not axes[1, 2]:
                axis.axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{index}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
