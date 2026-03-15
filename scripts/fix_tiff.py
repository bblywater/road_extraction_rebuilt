from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/massachusetts")
    parser.add_argument("--quarantine", default="data/quarantine")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    root = (project_root / args.root).resolve()
    quarantine = (project_root / args.quarantine).resolve()
    quarantine.mkdir(parents=True, exist_ok=True)

    broken = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".tif", ".tiff"} or not path.is_file():
            continue
        if cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is None:
            broken.append(path)

    for path in broken:
        target = quarantine / path.name
        shutil.move(str(path), str(target))
    print({"broken_files": [str(path) for path in broken], "count": len(broken)})


if __name__ == "__main__":
    main()
