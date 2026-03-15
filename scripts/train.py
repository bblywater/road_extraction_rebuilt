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
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    config = load_config(config_path)
    config["_config_dir"] = project_root
    experiment_dir = project_root / config["output"]["root"] / config["output"]["experiment_name"]

    trainer = Trainer(config, experiment_dir)
    train_loader = build_dataloader(config, split="train", shuffle=True)
    val_loader = build_dataloader(config, split="val", shuffle=False)
    trainer.fit(train_loader, val_loader, resume_from=args.resume)


if __name__ == "__main__":
    main()
