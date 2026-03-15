from __future__ import annotations

import os
from pathlib import Path


IGNORE = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "venv", "env"}


def build_tree(root: Path, prefix: str = "") -> list[str]:
    lines: list[str] = []
    entries = [entry for entry in sorted(root.iterdir(), key=lambda item: item.name.lower()) if entry.name not in IGNORE]
    for index, entry in enumerate(entries):
        is_last = index == len(entries) - 1
        branch = "└── " if is_last else "├── "
        lines.append(f"{prefix}{branch}{entry.name}")
        if entry.is_dir():
            extension = "    " if is_last else "│   "
            lines.extend(build_tree(entry, prefix + extension))
    return lines


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    tree = [project_root.name, *build_tree(project_root)]
    text = "\n".join(tree)
    output_path = project_root / "project_structure.txt"
    output_path.write_text(text, encoding="utf-8")
    print(text)
    print({"saved_to": str(output_path), "cwd": os.getcwd()})


if __name__ == "__main__":
    main()
