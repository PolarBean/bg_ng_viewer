#!/usr/bin/env python3
"""Generate one Neuroglancer state per BrainGlobe atlas family."""

import argparse
import re
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="skip state files that already exist",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    atlases = highest_resolution_atlases(list(remote_versions()))
    jobs = [
        (atlas, OUTPUT_DIR / f"{atlas.rsplit('_', 1)[0]}.json") for atlas in atlases
    ]
    if args.missing_only:
        jobs = [(atlas, output) for atlas, output in jobs if not output.exists()]

    progress = tqdm(jobs, unit="atlas")
    for atlas, output in progress:
        progress.set_description(atlas)
        generate_s3_json(atlas, output)


if __name__ == "__main__":
    main()
