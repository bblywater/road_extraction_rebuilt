from __future__ import annotations

from dataclasses import dataclass

import torch


def prediction_to_binary(prediction: torch.Tensor, from_logits: bool = True, threshold: float = 0.5) -> torch.Tensor:
    if from_logits:
        prediction = torch.sigmoid(prediction)
    return (prediction > threshold).float()


def compute_batch_confusion(prediction: torch.Tensor, target: torch.Tensor, from_logits: bool = True) -> dict[str, torch.Tensor]:
    pred = prediction_to_binary(prediction, from_logits=from_logits)
    tgt = (target > 0.5).float()
    return {
        "tp": (pred * tgt).sum().detach(),
        "fp": (pred * (1 - tgt)).sum().detach(),
        "fn": ((1 - pred) * tgt).sum().detach(),
        "tn": ((1 - pred) * (1 - tgt)).sum().detach(),
    }


def confusion_to_metrics(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor) -> dict[str, float]:
    precision_tensor = (tp + 1e-6) / (tp + fp + 1e-6)
    recall_tensor = (tp + 1e-6) / (tp + fn + 1e-6)
    return {
        "iou": float(((tp + 1e-6) / (tp + fp + fn + 1e-6)).item()),
        "f1": float((((2 * precision_tensor * recall_tensor) + 1e-6) / (precision_tensor + recall_tensor + 1e-6)).item()),
        "precision": float(precision_tensor.item()),
        "recall": float(recall_tensor.item()),
        "accuracy": float(((tp + tn + 1e-6) / (tp + fp + fn + tn + 1e-6)).item()),
        "dice": float(((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).item()),
        "specificity": float(((tn + 1e-6) / (tn + fp + 1e-6)).item()),
        "tp": float(tp.item()),
        "fp": float(fp.item()),
        "fn": float(fn.item()),
        "tn": float(tn.item()),
    }


def compute_batch_metrics(prediction: torch.Tensor, target: torch.Tensor, from_logits: bool = True) -> dict[str, float]:
    confusion = compute_batch_confusion(prediction, target, from_logits=from_logits)
    return confusion_to_metrics(confusion["tp"], confusion["fp"], confusion["fn"], confusion["tn"])


@dataclass
class SegmentationMeter:
    loss_sum: torch.Tensor | None = None
    batches: int = 0
    tp: torch.Tensor | None = None
    fp: torch.Tensor | None = None
    fn: torch.Tensor | None = None
    tn: torch.Tensor | None = None

    def update(self, loss: torch.Tensor, confusion: dict[str, torch.Tensor]) -> None:
        detached_loss = loss.detach()
        self.loss_sum = detached_loss if self.loss_sum is None else self.loss_sum + detached_loss
        self.tp = confusion["tp"] if self.tp is None else self.tp + confusion["tp"]
        self.fp = confusion["fp"] if self.fp is None else self.fp + confusion["fp"]
        self.fn = confusion["fn"] if self.fn is None else self.fn + confusion["fn"]
        self.tn = confusion["tn"] if self.tn is None else self.tn + confusion["tn"]
        self.batches += 1

    def mean(self) -> dict[str, float]:
        if self.batches == 0 or self.loss_sum is None or self.tp is None or self.fp is None or self.fn is None or self.tn is None:
            return {"loss": 0.0}
        summary = confusion_to_metrics(self.tp, self.fp, self.fn, self.tn)
        summary["loss"] = float((self.loss_sum / self.batches).item())
        return summary
