from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from road_extraction.pipelines import evaluate_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment)
    if not experiment_dir.is_absolute():
        experiment_dir = (project_root / experiment_dir).resolve()
    summary = evaluate_experiment(experiment_dir, split=args.split)
    print(summary)


if __name__ == "__main__":
    main()
