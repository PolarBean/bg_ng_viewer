#!/usr/bin/env python3
"""One-off: merge the Kim developmental atlas variants into one state per family.

BrainGlobe ships each Kim reference modality as its own atlas (STP, LSFM/iDISCO
and the MRI contrasts), even though the variants of one developmental stage share
a coordinate space and an annotation set. This writes a single Neuroglancer state
per family, with the remaining modalities added as additional reference layers
that are switched off until you toggle them.

    python combine_kim_dev_states.py

Nothing in bg_ng_viewer.py is changed, and the per-variant state files are left
alone: the combined states land in new files named after the family.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import URLError

from bg_ng_viewer import (
    TEMPLATE,
    Atlas,
    generate_s3_state,
    http_url,
    load_remote_atlas,
    remote_template_range,
    remote_versions,
)

OUTPUT_DIR = Path(__file__).parent / "bg_ng_state_files"
FAMILY_PREFIX = "kim_dev_mouse"
RESOLUTION_SUFFIX = re.compile(r"_\d+(?:\.\d+)?um$")

# The modality to show by default, best first. Anything unlisted sorts after
# these, so a family of MRI contrasts alone still gets a sensible primary.
PREFERRED = ("stp", "lsfm", "idisco")


def retry(call, attempts: int = 4):
    """S3 occasionally resets a chunk request; sampling a range is worth retrying."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except (URLError, TimeoutError, ConnectionError) as error:
            if attempt == attempts:
                raise
            print(f"  retrying after {error}")
            time.sleep(2 * attempt)


def family_name(names: list[str]) -> str:
    """Shared part of the variant names, e.g. kim_dev_mouse_e11-5."""
    return os.path.commonprefix(names).rstrip("_-")


def modality(name: str, family: str) -> str:
    """Variant name reduced to what distinguishes it, e.g. mri-adc."""
    return RESOLUTION_SUFFIX.sub("", name).removeprefix(family).strip("_-")


def rank(atlas: Atlas, name: str, family: str) -> tuple:
    label = modality(name, family)
    preference = next(
        (index for index, key in enumerate(PREFERRED) if key in label), len(PREFERRED)
    )
    return (preference, float(atlas.manifest["resolution"][0]), name)


def kim_families(atlases: dict[str, Atlas]) -> dict[str, list[str]]:
    """Group variants by the coordinate space they are registered in."""
    spaces: dict[str, list[str]] = defaultdict(list)
    for name, atlas in atlases.items():
        spaces[atlas.manifest["coordinate_space"]["name"]].append(name)
    return {family_name(sorted(names)): sorted(names) for names in spaces.values()}


def combine(family: str, names: list[str], atlases: dict[str, Atlas]) -> dict:
    ordered = sorted(names, key=lambda name: rank(atlases[name], name, family))
    primary, siblings = ordered[0], ordered[1:]
    print(f"{family}: template={modality(primary, family)} + {len(siblings)} hidden")

    state = retry(lambda: generate_s3_state(primary))
    layers = state["layers"]
    references = []
    for name in siblings:
        url = http_url(atlases[name].relative("template", TEMPLATE))
        image_range = retry(lambda: remote_template_range(url))
        references.append(
            {
                "type": "image",
                "source": f"{url}/|zarr3:",
                "shaderControls": {"normalized": {"range": list(image_range)}},
                "visible": False,
                "name": modality(name, family),
            }
        )

    # Keep the images together, ahead of the annotation and hemisphere layers.
    images = sum(1 for layer in layers if layer["type"] == "image")
    state["layers"] = layers[:images] + references + layers[images:]
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    names = [name for name in remote_versions() if name.startswith(FAMILY_PREFIX)]
    atlases = {name: load_remote_atlas(name) for name in names}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for family, members in sorted(kim_families(atlases).items()):
        state = combine(family, members, atlases)
        output = args.output_dir / f"{family}.json"
        output.write_text(json.dumps(state, indent=2) + "\n")
        print(f"  wrote {output}")


if __name__ == "__main__":
    main()
