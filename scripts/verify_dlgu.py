from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from road_extraction.data import MassachusettsRoadsDataset
from road_extraction.models import DLGUNet
from road_extraction.training.losses import DiceBCELoss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/massachusetts")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patches", type=int, default=18)
    args = parser.parse_args()

    data_root = (project_root / args.data_root).resolve()
    dataset = MassachusettsRoadsDataset(root=data_root, split="train", include_masks=True)
    subset = Subset(dataset, range(min(args.patches, len(dataset))))
    loader = DataLoader(subset, batch_size=2, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DLGUNet().to(device)
    criterion = DiceBCELoss(from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(loader), 1)
        print({"epoch": epoch, "loss": avg_loss})
        if avg_loss < 0.01:
            break


if __name__ == "__main__":
    main()
