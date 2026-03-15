from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models import build_model
from ..utils.io import ensure_dir, save_json
from ..utils.plotting import plot_training_curves
from .losses import build_loss
from .metrics import SegmentationMeter, compute_batch_confusion


class Trainer:
    def __init__(self, config: dict, experiment_dir: str | Path) -> None:
        self.config = config
        self.experiment_dir = ensure_dir(experiment_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(config).to(self.device)
        self.criterion = build_loss(config)
        self.from_logits = config["loss"].get("from_logits", True)
        if isinstance(self.criterion, nn.BCEWithLogitsLoss) and self.criterion.pos_weight is not None:
            self.criterion.pos_weight = self.criterion.pos_weight.to(self.device)
        optimizer_name = config["training"].get("optimizer", "adam").lower()
        optimizer_kwargs = {
            "params": self.model.parameters(),
            "lr": config["training"]["learning_rate"],
            "weight_decay": config["training"].get("weight_decay", 0.0),
        }
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(**optimizer_kwargs)
        elif optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(**optimizer_kwargs)
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                **optimizer_kwargs,
                momentum=config["training"].get("momentum", 0.9),
                nesterov=config["training"].get("nesterov", True),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        scheduler_cfg = config["training"].get("scheduler", {})
        self.scheduler = None
        if scheduler_cfg.get("name", "").lower() == "steplr":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_cfg.get("step_size", 10),
                gamma=scheduler_cfg.get("gamma", 0.5),
            )
        if scheduler_cfg.get("name", "").lower() == "poly":
            total_epochs = max(1, int(config["training"]["epochs"]))
            power = float(scheduler_cfg.get("power", 0.9))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: (1.0 - min(epoch, total_epochs) / total_epochs) ** power,
            )
        self.use_amp = bool(config["training"].get("amp", False) and torch.cuda.is_available())
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.history = {
            "train_loss": [],
            "train_iou": [],
            "train_f1": [],
            "val_loss": [],
            "val_iou": [],
            "val_f1": [],
            "lr": [],
        }
        self.best_metric = float("-inf")
        self.aux_weight = float(config["model"].get("aux_weight", 0.0))

    def _split_outputs(self, outputs):
        if isinstance(outputs, dict):
            logits = outputs["logits"]
            aux_logits = outputs.get("aux_logits")
            return logits, aux_logits
        if isinstance(outputs, tuple):
            logits = outputs[0]
            aux_logits = outputs[1] if len(outputs) > 1 else None
            return logits, aux_logits
        return outputs, None

    def _compute_loss(self, logits: torch.Tensor, masks: torch.Tensor, aux_logits: torch.Tensor | None = None) -> torch.Tensor:
        loss = self.criterion(logits, masks)
        if aux_logits is None or self.aux_weight <= 0.0:
            return loss
        aux_target = torch.nn.functional.interpolate(masks, size=aux_logits.shape[-2:], mode="nearest")
        return loss + self.aux_weight * self.criterion(aux_logits, aux_target)

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        meter = SegmentationMeter()
        self.model.train(training)
        log_interval = int(self.config["training"].get("log_interval", 25))
        total_steps = len(loader)
        phase = "train" if training else "val"
        use_tqdm = sys.stdout.isatty()
        iterator = (
            tqdm(loader, leave=False, desc=phase, miniters=log_interval, mininterval=1.0)
            if use_tqdm
            else loader
        )
        for step_idx, batch in enumerate(iterator, start=1):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            if training:
                self.optimizer.zero_grad()
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    logits, aux_logits = self._split_outputs(outputs)
                    loss = self._compute_loss(logits, masks, aux_logits)
            else:
                outputs = self.model(images)
                logits, aux_logits = self._split_outputs(outputs)
                loss = self._compute_loss(logits, masks, aux_logits)
            if training:
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config["training"].get("gradient_clip", 1.0),
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config["training"].get("gradient_clip", 1.0),
                    )
                    self.optimizer.step()
            confusion = compute_batch_confusion(logits.detach(), masks, from_logits=self.from_logits)
            meter.update(loss, confusion)
            if step_idx % log_interval == 0 or step_idx == total_steps:
                mean = meter.mean()
                if use_tqdm:
                    iterator.set_postfix(loss=f"{mean['loss']:.4f}", iou=f"{mean.get('iou', 0.0):.4f}")
                else:
                    print(
                        f"{phase}_step={step_idx}/{total_steps} "
                        f"loss={mean['loss']:.4f} iou={mean.get('iou', 0.0):.4f}"
                    )
        return meter.mean()

    def _save_checkpoint(self, epoch: int, best: bool = False) -> None:
        target = self.experiment_dir / ("best_model.pth" if best else f"checkpoint_epoch_{epoch}.pth")
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_metric": self.best_metric,
            "config": self.config,
        }
        torch.save(payload, target)

    def save_config(self) -> None:
        save_json(self.experiment_dir / "config.json", self.config)

    def resume(self, checkpoint_path: str | Path) -> int:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_metric = checkpoint.get("best_metric", float("-inf"))
        return int(checkpoint.get("epoch", 0)) + 1

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, resume_from: Optional[str | Path] = None) -> dict[str, list[float]]:
        self.save_config()
        start_epoch = 1
        if resume_from is not None:
            start_epoch = self.resume(resume_from)
        if self.device.type == "cuda":
            device_name = torch.cuda.get_device_name(self.device)
            print(f"device=cuda gpu={device_name} amp={self.use_amp}")
        else:
            print(f"device=cpu amp={self.use_amp}")
        total_epochs = self.config["training"]["epochs"]
        metric_name = self.config["output"].get("metric", "iou")
        save_every = self.config["output"].get("save_every", 5)
        early_cfg = self.config["training"].get("early_stopping", {})
        patience = early_cfg.get("patience", 8)
        min_delta = early_cfg.get("min_delta", 0.001)
        stale_epochs = 0

        for epoch in range(start_epoch, total_epochs + 1):
            start_time = time.time()
            train_metrics = self._run_epoch(train_loader, training=True)
            with torch.no_grad():
                val_metrics = self._run_epoch(val_loader, training=False)
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step()

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_iou"].append(train_metrics["iou"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_iou"].append(val_metrics["iou"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["lr"].append(current_lr)

            print(
                f"epoch={epoch}/{total_epochs} "
                f"train_loss={train_metrics['loss']:.4f} train_iou={train_metrics['iou']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_iou={val_metrics['iou']:.4f} "
                f"lr={current_lr:.6f} time={time.time() - start_time:.1f}s"
            )

            metric_value = val_metrics[metric_name]
            if metric_value > self.best_metric + min_delta:
                self.best_metric = metric_value
                stale_epochs = 0
                self._save_checkpoint(epoch, best=True)
            else:
                stale_epochs += 1
            if epoch % save_every == 0:
                self._save_checkpoint(epoch, best=False)
            save_json(self.experiment_dir / "training_history.json", self.history)
            plot_training_curves(self.history, self.experiment_dir / "training_curves.png")
            if stale_epochs >= patience:
                break
        return self.history
