from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from road_extraction.data import build_dataloader
from road_extraction.training import Trainer
from road_extraction.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    experiment_dir = (project_root / args.experiment).resolve()
    checkpoint_path = (project_root / args.checkpoint).resolve()
    config = load_config(experiment_dir / "config.json")
    config["_config_dir"] = project_root
    trainer = Trainer(config, experiment_dir)
    train_loader = build_dataloader(config, split="train", shuffle=True)
    val_loader = build_dataloader(config, split="val", shuffle=False)
    trainer.fit(train_loader, val_loader, resume_from=checkpoint_path)


if __name__ == "__main__":
    main()
