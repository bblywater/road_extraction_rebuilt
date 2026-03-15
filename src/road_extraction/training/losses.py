from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, from_logits: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            prediction = torch.sigmoid(prediction)
        prediction = prediction.contiguous().view(prediction.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        intersection = (prediction * target).sum(dim=1)
        union = prediction.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(
        self,
        from_logits: bool = True,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        pos = torch.tensor([pos_weight], dtype=torch.float32)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos) if from_logits else nn.BCELoss()
        self.dice = DiceLoss(from_logits=from_logits)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(self.bce, nn.BCEWithLogitsLoss):
            self.bce.pos_weight = self.bce.pos_weight.to(prediction.device)
        return self.bce_weight * self.bce(prediction, target) + self.dice_weight * self.dice(prediction, target)


class BinaryFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        from_logits: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.float()
        target = target.float()
        if self.from_logits:
            bce = nn.functional.binary_cross_entropy_with_logits(prediction, target, reduction="none")
            pt = torch.exp(-bce)
        else:
            prediction = prediction.clamp(min=1e-6, max=1.0 - 1e-6)
            bce = nn.functional.binary_cross_entropy(prediction, target, reduction="none")
            pt = prediction * target + (1.0 - prediction) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = alpha_t * ((1.0 - pt) ** self.gamma) * bce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_loss(config: dict) -> nn.Module:
    loss_cfg = config["loss"]
    name = loss_cfg["name"].lower()
    if name == "dice":
        return DiceLoss(from_logits=loss_cfg.get("from_logits", True))
    if name == "bce_with_logits":
        pos = torch.tensor([loss_cfg.get("pos_weight", 1.0)], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pos)
    if name == "dice_bce":
        return DiceBCELoss(
            from_logits=loss_cfg.get("from_logits", True),
            bce_weight=loss_cfg.get("bce_weight", 1.0),
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            pos_weight=loss_cfg.get("pos_weight", 1.0),
        )
    if name == "focal":
        return BinaryFocalLoss(
            alpha=loss_cfg.get("alpha", 0.25),
            gamma=loss_cfg.get("gamma", 2.0),
            from_logits=loss_cfg.get("from_logits", True),
            reduction=loss_cfg.get("reduction", "mean"),
        )
    raise ValueError(f"Unsupported loss: {name}")
