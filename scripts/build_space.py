"""Build an allowlisted Space directory without local data or credentials."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = {
    "huggingface-space/README.md": "README.md",
    "huggingface-space/app.py": "app.py",
    "huggingface-space/requirements.txt": "requirements.txt",
    "pyproject.toml": "pyproject.toml",
    "config.example.toml": "config.example.toml",
}


def build(target: Path) -> list[Path]:
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    files = {**PUBLIC_FILES}
    for source in sorted((ROOT / "gaia_agent").rglob("*.py")):
        files[source.relative_to(ROOT).as_posix()] = source.relative_to(ROOT).as_posix()
    target.mkdir(parents=True)
    copied = []
    try:
        for source_name, target_name in files.items():
            source = ROOT / source_name
            destination = target / target_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(destination)
    except Exception:
        # This call created the target; remove only its own incomplete bundle.
        shutil.rmtree(target)
        raise
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "space-bundle")
    args = parser.parse_args()
    copied = build(args.output)
    print(f"Built {len(copied)} public files in {args.output.resolve()}")


if __name__ == "__main__":
    main()
