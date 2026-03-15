from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from road_extraction.data import build_dataloader
from road_extraction.models import build_model
from road_extraction.training.losses import build_loss
from road_extraction.utils import load_config


def split_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"], outputs.get("aux_logits")
    if isinstance(outputs, tuple):
        return outputs[0], outputs[1] if len(outputs) > 1 else None
    return outputs, None


def run_step(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    device: torch.device,
    use_amp: bool,
) -> None:
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            logits, aux_logits = split_outputs(outputs)
            loss = criterion(logits, masks)
            if aux_logits is not None:
                aux_target = torch.nn.functional.interpolate(masks, size=aux_logits.shape[-2:], mode="nearest")
                loss = loss + float(getattr(model, "aux_weight", 0.0)) * criterion(aux_logits, aux_target)
    else:
        outputs = model(images)
        logits, aux_logits = split_outputs(outputs)
        loss = criterion(logits, masks)
        if aux_logits is not None:
            aux_target = torch.nn.functional.interpolate(masks, size=aux_logits.shape[-2:], mode="nearest")
            loss = loss + float(getattr(model, "aux_weight", 0.0)) * criterion(aux_logits, aux_target)
    loss.backward()
    optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", choices=["true", "false"], default=None)
    parser.add_argument("--amp", choices=["true", "false"], default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    config = load_config(config_path)
    config["_config_dir"] = project_root
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.pin_memory is not None:
        config["training"]["pin_memory"] = args.pin_memory == "true"
    if args.amp is not None:
        config["training"]["amp"] = args.amp == "true"

    loader = build_dataloader(config, split=args.split, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    criterion = build_loss(config)
    if isinstance(criterion, torch.nn.BCEWithLogitsLoss) and criterion.pos_weight is not None:
        criterion.pos_weight = criterion.pos_weight.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    use_amp = bool(config["training"].get("amp", False) and device.type == "cuda")

    iterator = iter(loader)
    for _ in range(args.warmup):
        batch = next(iterator)
        run_step(model, criterion, optimizer, batch, device, use_amp)
    if device.type == "cuda":
        torch.cuda.synchronize()

    wait_time = 0.0
    step_time = 0.0
    total_start = time.perf_counter()
    for _ in range(args.batches):
        wait_start = time.perf_counter()
        batch = next(iterator)
        wait_time += time.perf_counter() - wait_start

        step_start = time.perf_counter()
        run_step(model, criterion, optimizer, batch, device, use_amp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_time += time.perf_counter() - step_start
    total_time = time.perf_counter() - total_start

    print(
        {
            "device": str(device),
            "split": args.split,
            "warmup_batches": args.warmup,
            "measured_batches": args.batches,
            "wait_time_sec": round(wait_time, 3),
            "step_time_sec": round(step_time, 3),
            "total_time_sec": round(total_time, 3),
            "sec_per_batch": round(total_time / args.batches, 4),
            "batches_per_sec": round(args.batches / total_time, 3),
        }
    )


if __name__ == "__main__":
    main()
