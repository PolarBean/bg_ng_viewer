#!/usr/bin/env python3
"""Generate one Neuroglancer state per BrainGlobe atlas family.

Each state is written as ``<family>_v<version>.json`` and copied to
``<family>.json``, which always holds the latest version so links to it are
stable.
"""

import argparse
import re
import shutil
from pathlib import Path

from tqdm import tqdm

from bg_ng_viewer import generate_s3_json, remote_versions

OUTPUT_DIR = Path(__file__).parent / "bg_ng_state_files"
RESOLUTION_SUFFIX = re.compile(r"^(.*)_(\d+(?:\.\d+)?)um$")


def highest_resolution_atlases(names: list[str]) -> list[str]:
    """Keep the smallest voxel size for each atlas family."""
    selected: dict[str, tuple[float, str]] = {}
    for name in names:
        match = RESOLUTION_SUFFIX.fullmatch(name)
        family, resolution = (
            (match.group(1), float(match.group(2))) if match else (name, float("inf"))
        )
        if family not in selected or resolution < selected[family][0]:
            selected[family] = (resolution, name)
    return sorted(name for _, name in selected.values())


def state_paths(atlas: str, version: str, directory: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    """(versioned, latest) state files for an atlas."""
    family = atlas.rsplit("_", 1)[0]
    return directory / f"{family}_v{version}.json", directory / f"{family}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="only generate states for atlas versions without a versioned file",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    versions = remote_versions()
    jobs = [
        (atlas, versions[atlas], *state_paths(atlas, versions[atlas]))
        for atlas in highest_resolution_atlases(list(versions))
    ]

    progress = tqdm(jobs, unit="atlas")
    for atlas, version, versioned, latest in progress:
        progress.set_description(atlas)
        if not (args.missing_only and versioned.exists()):
            generate_s3_json(atlas, versioned, version=version)
        shutil.copyfile(versioned, latest)


if __name__ == "__main__":
    main()
