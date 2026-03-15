from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from ..data import MassachusettsRoadsDataset
from ..training.metrics import compute_batch_metrics
from ..utils.io import ensure_dir, load_config, save_json
from ..utils.plotting import plot_evaluation_summary, plot_sample_visuals
from .inference import load_experiment_model, predict_single_image


def _results_dir_name(split: str) -> str:
    return "full_evaluation" if split == "test" else f"full_evaluation_{split}"


def evaluate_experiment(experiment_dir: str | Path, split: str = "test") -> dict:
    experiment_dir = Path(experiment_dir)
    config = load_config(experiment_dir / "config.json")
    config["_config_dir"] = experiment_dir.parent.parent if (experiment_dir / "config.json").exists() else Path(".")
    model, _, checkpoint = load_experiment_model(experiment_dir)
    dataset = MassachusettsRoadsDataset(
        root=(Path(config["dataset"]["root"]) if Path(config["dataset"]["root"]).is_absolute() else (experiment_dir.parent.parent / config["dataset"]["root"]).resolve()),
        split=split,
        patch_size=config["dataset"].get("patch_size", 512),
        target_size=config["dataset"].get("target_size", 1536),
        corrupted_files=config["dataset"].get("corrupted_files", []),
        include_masks=True,
    )
    results_dir = ensure_dir(experiment_dir / _results_dir_name(split))

    rows = []
    metrics_all = []
    sample_visuals = []
    for index in range(len(dataset.records)):
        sample = dataset.get_full_sample(index)
        prediction = predict_single_image(
            model,
            sample["image_path"],
            patch_size=config["dataset"].get("patch_size", 512),
            target_size=config["dataset"].get("target_size", 1536),
        )
        mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        pred_tensor = torch.from_numpy(prediction["prediction"]).unsqueeze(0).unsqueeze(0)
        prob_tensor = torch.from_numpy(prediction["probability"]).unsqueeze(0).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        metrics = compute_batch_metrics(prob_tensor, mask_tensor, from_logits=False)
        rows.append(
            {
                "sample_id": Path(sample["image_path"]).stem,
                "iou": metrics["iou"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accuracy": metrics["accuracy"],
                "dice": metrics["dice"],
            }
        )
        metrics_all.append(metrics)
        if len(sample_visuals) < 5:
            error = np.zeros((*mask.shape, 3), dtype=np.float32)
            tp = (prediction["prediction"] == 1) & (mask == 1)
            fp = (prediction["prediction"] == 1) & (mask == 0)
            fn = (prediction["prediction"] == 0) & (mask == 1)
            error[tp] = [0, 1, 0]
            error[fp] = [1, 0, 0]
            error[fn] = [0, 0, 1]
            sample_visuals.append(
                {
                    "image": prediction["image"],
                    "mask": mask,
                    "probability": prediction["probability"],
                    "prediction": prediction["prediction"],
                    "error": error,
                    "iou": metrics["iou"],
                    "f1": metrics["f1"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "sample_results.csv", index=False, encoding="utf-8-sig")

    mean_metrics = {key: float(np.mean([metric[key] for metric in metrics_all])) for key in ("iou", "f1", "precision", "recall", "accuracy", "dice")}
    std_metrics = {key: float(np.std([metric[key] for metric in metrics_all])) for key in ("iou", "f1", "precision", "recall")}
    confusion = np.array(
        [
            [sum(metric["tn"] for metric in metrics_all), sum(metric["fp"] for metric in metrics_all)],
            [sum(metric["fn"] for metric in metrics_all), sum(metric["tp"] for metric in metrics_all)],
        ]
    )

    summary = {
        "total_samples": int(len(metrics_all)),
        "average_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "range_metrics": {
            "iou_min": float(df["iou"].min()),
            "iou_max": float(df["iou"].max()),
        },
        "confusion_matrix": {
            "tp": int(sum(metric["tp"] for metric in metrics_all)),
            "fp": int(sum(metric["fp"] for metric in metrics_all)),
            "fn": int(sum(metric["fn"] for metric in metrics_all)),
            "tn": int(sum(metric["tn"] for metric in metrics_all)),
        },
        "model_info": {
            "epoch": int(checkpoint.get("epoch", 0)),
            "best_metric": float(checkpoint.get("best_metric", 0.0)),
            "parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        },
        "dataset_info": {
            "root": str(config["dataset"]["root"]),
            "split": split,
            "patch_size": int(config["dataset"].get("patch_size", 512)),
        },
    }
    save_json(results_dir / "summary_results.json", summary)
    plot_evaluation_summary(df["iou"], df["precision"], df["recall"], mean_metrics, confusion, results_dir)
    plot_sample_visuals(sample_visuals, results_dir / "sample_visualizations")

    report_lines = [
        "# Evaluation Report",
        "",
        f"- Samples: {summary['total_samples']}",
        f"- Mean IoU: {mean_metrics['iou']:.4f}",
        f"- Mean F1: {mean_metrics['f1']:.4f}",
        f"- Mean Precision: {mean_metrics['precision']:.4f}",
        f"- Mean Recall: {mean_metrics['recall']:.4f}",
        "",
        "## Model",
        f"- Epoch: {summary['model_info']['epoch']}",
        f"- Best Metric: {summary['model_info']['best_metric']:.4f}",
        f"- Parameters: {summary['model_info']['parameters']}",
    ]
    (results_dir / "evaluation_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return summary
