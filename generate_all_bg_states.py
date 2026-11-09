#!/usr/bin/env python3
"""Generate Neuroglancer states for every published BrainGlobe atlas version.

Each version is written as ``<family>_v<version>.json``; the latest version is
also copied to ``<family>.json`` so links to it are stable.
"""

import argparse
import re
import shutil
from pathlib import Path

from tqdm import tqdm

from bg_ng_viewer import generate_s3_json, published_versions, remote_versions

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
    atlases = highest_resolution_atlases(list(versions))
    jobs = [
        (atlas, version)
        for atlas in tqdm(atlases, desc="listing versions", unit="atlas")
        for version in published_versions(atlas)
    ]

    progress = tqdm(jobs, unit="state")
    for atlas, version in progress:
        progress.set_description(f"{atlas} v{version}")
        versioned, latest = state_paths(atlas, version)
        if not (args.missing_only and versioned.exists()):
            generate_s3_json(atlas, versioned, version=version)
        if version == versions[atlas]:
            shutil.copyfile(versioned, latest)


if __name__ == "__main__":
    main()
