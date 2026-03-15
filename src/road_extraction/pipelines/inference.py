from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from ..data.datasets import crop_back, merge_tiles, normalize_image, pad_to_size, tile_image
from ..models import build_model
from ..utils.io import load_config


def _extract_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def load_experiment_model(experiment_dir: str | Path, checkpoint_name: str = "best_model.pth"):
    experiment_dir = Path(experiment_dir)
    config = load_config(experiment_dir / "config.json")
    model = build_model(config)
    checkpoint_path = experiment_dir / checkpoint_name
    if not checkpoint_path.exists():
        checkpoint_path = experiment_dir / "checkpoints" / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.eval()
    return model, config, checkpoint


def predict_single_image(
    model,
    image_path: str | Path,
    patch_size: int = 512,
    target_size: int = 1536,
    threshold: float = 0.5,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray | float | tuple[int, int]]:
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    padded, pad = pad_to_size(image, target_size, is_mask=False)
    tiles = tile_image(padded, patch_size)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    predictions: list[tuple[tuple[int, int], np.ndarray]] = []

    with torch.no_grad():
        for coords, tile in tiles:
            tensor = torch.from_numpy(normalize_image(tile)).unsqueeze(0).float().to(device)
            logits = _extract_logits(model(tensor))
            probability = torch.sigmoid(logits).cpu().numpy()[0, 0]
            predictions.append((coords, probability))

    probability_map = merge_tiles(predictions, target_size, patch_size)
    probability_map = crop_back(probability_map, pad, original_shape)
    binary_mask = (probability_map > threshold).astype(np.float32)
    return {
        "image": image,
        "probability": probability_map,
        "prediction": binary_mask,
        "coverage": float(binary_mask.mean()),
        "road_pixels": float(binary_mask.sum()),
        "original_shape": original_shape,
    }
