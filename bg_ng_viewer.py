#!/usr/bin/env python3
"""Generate or serve a BrainGlobe atlas state for Neuroglancer."""

from __future__ import annotations

import argparse
import configparser
import csv
import gzip
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError
from urllib.request import urlopen

STORE_NAME = "brainglobe-atlasapi"
S3_ROOT = "brainglobe/atlas-rc2"
HTTP_ROOT = "https://brainglobe.s3.us-west-2.amazonaws.com/atlas-rc2"
TEMPLATE = "template.ome.zarr"
ANNOTATION = "annotations_compressed.ome.zarr"
HEMISPHERES = "hemispheres.ome.zarr"
PRECOMPUTED = "annotations.precomputed"
TERMINOLOGY = "terminology.csv"
MANIFEST = "manifest.json"

# Assumed cross-section panel size in pixels, used to fit the atlas in the 2D views.
PANEL_PIXELS = 600

# How many region meshes are shown when the state is first opened.
DEFAULT_MESHES = 40

# BrainGlobe orientation letters name where an axis starts, so increasing indexes
# travel towards the opposite pole.
AXIS_DIRECTIONS = {
    "a": "posterior",
    "p": "anterior",
    "s": "inferior",
    "i": "superior",
    "r": "left",
    "l": "right",
}
OPPOSITE = {
    "posterior": "anterior",
    "anterior": "posterior",
    "inferior": "superior",
    "superior": "inferior",
    "left": "right",
    "right": "left",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def http_url(relative: str) -> str:
    return f"{HTTP_ROOT}/{relative.lstrip('/')}"


@cache
def http_text(url: str) -> str:
    with urlopen(url, timeout=30) as response:
        return response.read().decode()


def component_relative(component: dict, filename: str = "") -> str:
    directory = component["location"].strip("/")
    return f"{directory}/{filename}".rstrip("/")


def reference_name(reference: dict) -> str:
    """Layer name for an additional reference, without the redundant suffix."""
    return reference["name"].removesuffix("-template")


@dataclass(frozen=True)
class Atlas:
    manifest: dict
    store: Path | None = None
    manifest_file: Path | None = None

    def relative(self, component: str, filename: str = "") -> str:
        return component_relative(self.manifest[component], filename)

    @property
    def references(self) -> list[dict]:
        return self.manifest.get("additional_references") or []

    def path(self, component: str, filename: str = "") -> Path:
        if self.store is None:
            raise ValueError("Atlas is not local")
        return self.store / self.relative(component, filename)


def atlas_store(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    return root if (root / "atlases").is_dir() else root / STORE_NAME


def load_local_atlas(root: str | Path, name: str) -> Atlas | None:
    store = atlas_store(root)
    manifests = list((store / "atlases" / name).glob(f"*/{MANIFEST}"))
    if not manifests:
        return None

    def version(path: Path) -> tuple[int, ...]:
        return tuple(
            int(part) for part in path.parent.name.replace(".", "_").split("_")
        )

    manifest_file = max(manifests, key=version)
    return Atlas(read_json(manifest_file), store, manifest_file)


@cache
def remote_versions() -> dict[str, str]:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(http_text(http_url("atlases/last_versions.conf")))
    return dict(parser["atlases"])


def load_remote_atlas(name: str) -> Atlas:
    versions = remote_versions()
    if name not in versions:
        raise SystemExit(f"Atlas {name!r} is not on BrainGlobe S3")
    version = versions[name].replace(".", "_")
    url = http_url(f"atlases/{name}/{version}/{MANIFEST}")
    return Atlas(json.loads(http_text(url)))


def load_structures(atlas: Atlas) -> list[dict[str, str]]:
    relative = atlas.relative("terminology", TERMINOLOGY)
    local = atlas.store / relative if atlas.store else None
    text = (
        local.read_text()
        if local and local.is_file()
        else http_text(http_url(relative))
    )
    return list(csv.DictReader(text.splitlines()))


def multiscale(metadata: dict) -> dict:
    return metadata["attributes"]["ome"]["multiscales"][0]


def find_scale(zarr_path: Path, resolution_um: float) -> str | None:
    if not (zarr_path / "zarr.json").is_file():
        return None
    for dataset in multiscale(read_json(zarr_path / "zarr.json"))["datasets"]:
        scale_um = dataset["coordinateTransformations"][0]["scale"][0] * 1000
        if abs(scale_um - resolution_um) < 1e-6:
            return dataset["path"]
    return None


def atlas_scale(zarr_path: Path, resolution_um: float) -> str:
    scale = find_scale(zarr_path, resolution_um)
    if scale is None:
        raise SystemExit(f"No {resolution_um:g} µm scale in {zarr_path}")
    return scale


def has_chunks(scale_path: Path) -> bool:
    chunk_dir = scale_path / "c"
    return chunk_dir.is_dir() and any(path.is_file() for path in chunk_dir.rglob("*"))


def chunk_key(metadata: dict, coordinate: tuple[int, ...]) -> str:
    encoding = metadata["chunk_key_encoding"]
    parts = [str(value) for value in coordinate]
    if encoding["name"] == "default":
        parts.insert(0, "c")
    elif encoding["name"] != "v2":
        raise ValueError(f"Unsupported Zarr chunk encoding: {encoding['name']}")
    return encoding["configuration"]["separator"].join(parts)


def decode_chunk(data: bytes | None, metadata: dict):
    import numpy as np
    import zstandard

    shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
    dtype = np.dtype(metadata["data_type"])
    if data is None:
        return np.full(shape, metadata["fill_value"], dtype=dtype)

    endian = None
    for codec in reversed(metadata["codecs"]):
        name = codec["name"]
        if name == "zstd":
            data = zstandard.ZstdDecompressor().decompress(data)
        elif name == "gzip":
            data = gzip.decompress(data)
        elif name == "bytes":
            endian = codec.get("configuration", {}).get("endian")
        else:
            raise ValueError(f"Unsupported Zarr codec: {name}")

    if endian:
        dtype = dtype.newbyteorder("<" if endian == "little" else ">")
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def sampled_zarr_range(metadata: dict, read_chunk, workers: int = 1) -> list[float]:
    """Return min/max from up to 16 evenly distributed chunks."""
    import numpy as np

    shape = np.asarray(metadata["shape"], dtype=int)
    chunks = np.asarray(
        metadata["chunk_grid"]["configuration"]["chunk_shape"], dtype=int
    )
    grid = (shape + chunks - 1) // chunks
    count = int(np.prod(grid))
    indexes = set(np.linspace(0, count - 1, min(16, count), dtype=int))
    indexes.add(int(np.ravel_multi_index(grid // 2, grid)))

    def chunk_range(index: int) -> tuple[float, float]:
        coordinate = np.unravel_index(index, tuple(grid))
        chunk = decode_chunk(read_chunk(coordinate), metadata)
        return float(chunk.min()), float(chunk.max())

    if workers == 1:
        ranges = [chunk_range(index) for index in sorted(indexes)]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(indexes))) as pool:
            ranges = list(pool.map(chunk_range, sorted(indexes)))
    return [min(low for low, _ in ranges), max(high for _, high in ranges)]


@cache
def local_scale_range(scale_path: str) -> tuple[float, float]:
    path = Path(scale_path)
    metadata = read_json(path / "zarr.json")

    def read_chunk(coordinate) -> bytes | None:
        file = path / chunk_key(metadata, coordinate)
        return file.read_bytes() if file.is_file() else None

    return tuple(sampled_zarr_range(metadata, read_chunk, workers=16))


@cache
def remote_template_range(template_url: str) -> tuple[float, float]:
    root = template_url.rstrip("/")
    group = json.loads(http_text(f"{root}/zarr.json"))
    scale = multiscale(group)["datasets"][-1]["path"]
    scale_url = f"{root}/{scale}"
    metadata = json.loads(http_text(f"{scale_url}/zarr.json"))

    def read_chunk(coordinate) -> bytes | None:
        try:
            with urlopen(
                f"{scale_url}/{chunk_key(metadata, coordinate)}", timeout=30
            ) as response:
                return response.read()
        except HTTPError as error:
            if error.code == 404:
                return None
            raise

    return tuple(sampled_zarr_range(metadata, read_chunk, workers=16))


def local_zarr_range(atlas: Atlas, relative: str) -> tuple[float, float] | None:
    """Range from the coarsest locally available scale of an image, if any."""
    if atlas.store is None:
        return None
    image = atlas.store / relative
    metadata_file = image / "zarr.json"
    if not metadata_file.is_file():
        return None
    for dataset in reversed(multiscale(read_json(metadata_file))["datasets"]):
        scale_path = image / dataset["path"]
        if has_chunks(scale_path):
            return local_scale_range(str(scale_path))
    return None


def local_template_range(atlas: Atlas) -> tuple[float, float] | None:
    return local_zarr_range(atlas, atlas.relative("template", TEMPLATE))


def anatomical_directions(orientation: str) -> dict[str, list[float]] | None:
    """Map anatomical directions to unit vectors in Neuroglancer xyz."""
    if len(orientation) != 3 or not set(orientation) <= set(AXIS_DIRECTIONS):
        return None
    vectors: dict[str, list[float]] = {}
    # Neuroglancer xyz is the reverse of the BrainGlobe axis order.
    for index, letter in enumerate(reversed(orientation.lower())):
        axis = [0.0, 0.0, 0.0]
        axis[index] = 1.0
        towards = AXIS_DIRECTIONS[letter]
        vectors[towards] = axis
        vectors[OPPOSITE[towards]] = [-value for value in axis]
    return vectors if len(vectors) == 6 else None


def matrix_quaternion(right, down, forward) -> list[float]:
    """Quaternion (x, y, z, w) for the rotation whose columns are the screen axes."""
    m = list(zip(right, down, forward))
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        scale = (trace + 1.0) ** 0.5 * 2
        q = [
            (m[2][1] - m[1][2]) / scale,
            (m[0][2] - m[2][0]) / scale,
            (m[1][0] - m[0][1]) / scale,
            scale / 4,
        ]
    else:
        i = max(range(3), key=lambda axis: m[axis][axis])
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = (1.0 + m[i][i] - m[j][j] - m[k][k]) ** 0.5 * 2
        q = [0.0, 0.0, 0.0, (m[k][j] - m[j][k]) / scale]
        q[i] = scale / 4
        q[j] = (m[j][i] + m[i][j]) / scale
        q[k] = (m[k][i] + m[i][k]) / scale
    return q


def top_down_orientation(orientation: str) -> list[float] | None:
    """Look down the dorsoventral axis with anterior up and the right hemisphere right."""
    directions = anatomical_directions(orientation)
    if directions is None:
        return None
    forward = directions["inferior"]  # into the screen
    down = directions["posterior"]
    right = [
        down[1] * forward[2] - down[2] * forward[1],
        down[2] * forward[0] - down[0] * forward[2],
        down[0] * forward[1] - down[1] * forward[0],
    ]
    return matrix_quaternion(right, down, forward)


def structure_depth(row: dict[str, str]) -> int:
    """Depth of a structure in the hierarchy, with the root at 0."""
    path = (row.get("root_identifier_path") or "").strip()
    if not path:
        return 0
    return max(len(json.loads(path)) - 1, 0)


def default_segments(
    structures: list[dict[str, str]], limit: int = DEFAULT_MESHES
) -> set[int]:
    """Pick up to `limit` segments to show, filling the hierarchy top down.

    Whole levels are taken while they fit; the level that would overflow
    contributes a random sample of the remaining budget, and deeper levels are
    left out. The sample is seeded so regenerating a state is reproducible.
    """
    levels: dict[int, list[int]] = {}
    for row in structures:
        levels.setdefault(structure_depth(row), []).append(int(row["annotation_value"]))

    chosen: set[int] = set()
    generator = random.Random(0)
    for depth in sorted(levels):
        ids = sorted(levels[depth])
        remaining = limit - len(chosen)
        if remaining <= 0:
            break
        if len(ids) <= remaining:
            chosen.update(ids)
        else:
            chosen.update(generator.sample(ids, remaining))
            break
    return chosen


def build_state(
    manifest: dict,
    structures: list[dict[str, str]],
    sources: dict[str, str],
    image_range: tuple[float, float],
    references: Sequence[tuple[str, str, tuple[float, float]]] = (),
) -> dict:
    values = [int(row["annotation_value"]) for row in structures]
    colors = {
        str(int(row["annotation_value"])): row["color_hex_triplet"]
        for row in structures
    }
    # The annotation layer carries the volume only, so every region is painted in
    # the cross sections without any mesh being fetched for it.
    annotation_source: str | list = sources["annotation"]
    mesh_layers = []
    if "precomputed" in sources:
        annotation_source = [
            annotation_source,
            {
                "url": sources["precomputed"],
                "enableDefaultSubsources": False,
                "subsources": {"properties": True},
            },
        ]
        # A second layer holds the meshes alone - it has no volume, so it only
        # shows in 3D. "!" lists a segment without showing it, keeping the region
        # one tick away in the segments tab.
        meshed = default_segments(structures)
        mesh_layers.append(
            {
                "type": "segmentation",
                "source": {
                    "url": sources["precomputed"],
                    "enableDefaultSubsources": False,
                    "subsources": {"properties": True, "mesh": True},
                },
                "segments": [
                    str(value) if value in meshed else f"!{value}" for value in values
                ],
                "segmentColors": colors,
                "objectAlpha": 0.55,
                "tab": "segments",
                "name": "region meshes",
            }
        )

    shape = list(reversed(manifest["shape"]))
    resolution = list(reversed(manifest["resolution"]))
    projection_orientation = top_down_orientation(manifest.get("orientation", ""))
    return {
        "dimensions": {
            axis: [value * 1e-6, "m"] for axis, value in zip("xyz", resolution)
        },
        "position": [value / 2 for value in shape],
        "crossSectionScale": max(shape) / PANEL_PIXELS,
        **(
            {"projectionOrientation": projection_orientation}
            if projection_orientation
            else {}
        ),
        "projectionScale": max(shape) * 1.25,
        "layers": [
            {
                "type": "image",
                "source": sources["template"],
                "shaderControls": {"normalized": {"range": list(image_range)}},
                "volumeRendering": "off",
                "name": "template",
            },
            *(
                {
                    "type": "image",
                    "source": source,
                    "shaderControls": {"normalized": {"range": list(range_)}},
                    "visible": False,
                    "name": name,
                }
                for name, source, range_ in references
            ),
            {
                "type": "segmentation",
                "source": annotation_source,
                "segments": [str(value) for value in values],
                "segmentColors": colors,
                "objectAlpha": 0.55,
                "tab": "segments",
                "name": "annotation",
            },
            *mesh_layers,
            {
                "type": "segmentation",
                "source": sources["hemisphere"],
                "segments": ["1", "2"],
                "segmentColors": {"1": "#4c78a8", "2": "#f58518"},
                "visible": False,
                "name": "hemisphere mask",
            },
        ],
        "showSlices": False,
        "selectedLayer": {"visible": True, "layer": "annotation"},
        "layout": "4panel-alt",
    }


def generate_s3_state(
    atlas_name: str,
    brainglobe_dir: str | Path = Path.home() / ".brainglobe",
    image_range: tuple[float, float] | None = None,
) -> dict:
    atlas = load_local_atlas(brainglobe_dir, atlas_name) or load_remote_atlas(
        atlas_name
    )
    template = atlas.relative("template", TEMPLATE)
    annotation = atlas.relative("annotation_set", ANNOTATION)
    annotation_dir = atlas.relative("annotation_set")
    sources = {
        "template": f"{http_url(template)}/|zarr3:",
        "annotation": f"{http_url(annotation)}/|zarr3:",
        "hemisphere": f"{http_url(f'{annotation_dir}/{HEMISPHERES}')}/|zarr3:",
        "precomputed": f"precomputed://{http_url(f'{annotation_dir}/{PRECOMPUTED}')}",
    }
    if image_range is None:
        image_range = local_template_range(atlas) or remote_template_range(
            http_url(template)
        )
    references = []
    for reference in atlas.references:
        relative = component_relative(reference, TEMPLATE)
        references.append(
            (
                reference_name(reference),
                f"{http_url(relative)}/|zarr3:",
                local_zarr_range(atlas, relative)
                or remote_template_range(http_url(relative)),
            )
        )
    return build_state(
        atlas.manifest, load_structures(atlas), sources, image_range, references
    )


def generate_s3_json(
    atlas_name: str,
    output: str | Path,
    brainglobe_dir: str | Path = Path.home() / ".brainglobe",
    image_range: tuple[float, float] | None = None,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = generate_s3_state(atlas_name, brainglobe_dir, image_range)
    output.write_text(json.dumps(state, indent=2) + "\n")
    return output


def sync_file(fs, remote: str, local: Path, label: str) -> None:
    size = fs.info(remote)["size"]
    if local.is_file() and local.stat().st_size == size:
        return
    print(f"Downloading {label}...")
    local.parent.mkdir(parents=True, exist_ok=True)
    fs.get(remote, local)


def sync_tree(fs, remote: str, local: Path, label: str) -> None:
    files = [
        (path, local / Path(path).relative_to(remote), info["size"])
        for path, info in fs.find(remote, detail=True).items()
        if info.get("type") != "directory"
    ]
    missing = [
        (remote_file, local_file)
        for remote_file, local_file, size in files
        if not local_file.is_file() or local_file.stat().st_size != size
    ]
    if not missing:
        return
    print(f"Downloading {len(missing)} {label} files...")

    def download(item) -> None:
        remote_file, local_file = item
        local_file.parent.mkdir(parents=True, exist_ok=True)
        fs.get(remote_file, local_file)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(download, missing))


def validate_local_atlas(atlas: Atlas) -> str:
    required = [
        atlas.path("template", f"{TEMPLATE}/zarr.json"),
        atlas.path("annotation_set", f"{ANNOTATION}/zarr.json"),
        atlas.path("annotation_set", f"{HEMISPHERES}/zarr.json"),
        atlas.path("terminology", TERMINOLOGY),
        atlas.path("coordinate_space", MANIFEST),
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Atlas is missing:\n" + "\n".join(map(str, missing)))

    scale = atlas_scale(
        atlas.path("template", TEMPLATE), float(atlas.manifest["resolution"][0])
    )
    for label, path in (
        ("template", atlas.path("template", f"{TEMPLATE}/{scale}")),
        ("annotation", atlas.path("annotation_set", f"{ANNOTATION}/{scale}")),
        ("hemisphere", atlas.path("annotation_set", f"{HEMISPHERES}/{scale}")),
    ):
        if not has_chunks(path):
            raise SystemExit(f"{label.capitalize()} chunks are missing from {path}")
    return scale


def ensure_local_data(atlas: Atlas) -> str:
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    assert atlas.store is not None and atlas.manifest_file is not None
    manifest_relative = atlas.manifest_file.relative_to(atlas.store).as_posix()
    if not fs.exists(f"{S3_ROOT}/{manifest_relative}"):
        return validate_local_atlas(atlas)

    files = [
        (atlas.relative("template", f"{TEMPLATE}/zarr.json"), "template metadata"),
        (
            atlas.relative("annotation_set", f"{ANNOTATION}/zarr.json"),
            "annotation metadata",
        ),
        (
            atlas.relative("annotation_set", f"{HEMISPHERES}/zarr.json"),
            "hemisphere metadata",
        ),
        (atlas.relative("terminology", TERMINOLOGY), "terminology"),
        (atlas.relative("coordinate_space", MANIFEST), "coordinate metadata"),
    ]
    for relative, label in files:
        sync_file(fs, f"{S3_ROOT}/{relative}", atlas.store / relative, label)

    scale = atlas_scale(
        atlas.path("template", TEMPLATE), float(atlas.manifest["resolution"][0])
    )
    trees = [
        (atlas.relative("template", f"{TEMPLATE}/{scale}"), "template chunks"),
        (
            atlas.relative("annotation_set", f"{ANNOTATION}/{scale}"),
            "annotation chunks",
        ),
        (
            atlas.relative("annotation_set", f"{HEMISPHERES}/{scale}"),
            "hemisphere chunks",
        ),
        (atlas.relative("annotation_set", f"{PRECOMPUTED}/mesh"), "mesh"),
        (
            atlas.relative("annotation_set", f"{PRECOMPUTED}/segment_properties"),
            "segment properties",
        ),
    ]
    sync_file(
        fs,
        f"{S3_ROOT}/{atlas.relative('annotation_set', f'{PRECOMPUTED}/info')}",
        atlas.path("annotation_set", f"{PRECOMPUTED}/info"),
        "precomputed metadata",
    )
    for reference in atlas.references:
        name = reference_name(reference)
        image = component_relative(reference, TEMPLATE)
        metadata = f"{image}/zarr.json"
        if not fs.exists(f"{S3_ROOT}/{metadata}"):
            print(f"Skipping {name}: not on BrainGlobe S3")
            continue
        sync_file(
            fs, f"{S3_ROOT}/{metadata}", atlas.store / metadata, f"{name} metadata"
        )
        reference_scale = find_scale(
            atlas.store / image, float(atlas.manifest["resolution"][0])
        )
        if reference_scale is None:
            print(f"Skipping {name}: no matching scale")
            continue
        trees.append((f"{image}/{reference_scale}", f"{name} chunks"))

    for relative, label in trees:
        sync_tree(fs, f"{S3_ROOT}/{relative}", atlas.store / relative, label)
    return scale


def local_references(atlas: Atlas) -> list[tuple[str, Path, str]]:
    """Additional references whose chunks are available in the local store."""
    assert atlas.store is not None
    resolution = float(atlas.manifest["resolution"][0])
    found = []
    for reference in atlas.references:
        path = atlas.store / component_relative(reference, TEMPLATE)
        scale = find_scale(path, resolution)
        if scale and has_chunks(path / scale):
            found.append((reference_name(reference), path, scale))
    return found


def make_local_state(
    atlas: Atlas, scale: str, origin: str
) -> tuple[dict, dict[str, tuple[Path, str]]]:
    mounts = {
        "template": (atlas.path("template", TEMPLATE), scale),
        "annotation": (atlas.path("annotation_set", ANNOTATION), scale),
        "hemisphere": (atlas.path("annotation_set", HEMISPHERES), scale),
    }
    sources = {name: f"{origin}/ng/{name}/|zarr3:" for name in mounts}
    precomputed = atlas.path("annotation_set", PRECOMPUTED)
    if (precomputed / "info").is_file():
        sources["precomputed"] = (
            f"precomputed://{origin}/{precomputed.relative_to(atlas.store).as_posix()}"
        )

    references = []
    for name, path, reference_scale in local_references(atlas):
        mounts[name] = (path, reference_scale)
        references.append(
            (
                name,
                f"{origin}/ng/{name}/|zarr3:",
                local_scale_range(str(path / reference_scale)),
            )
        )

    image_range = local_scale_range(str(mounts["template"][0] / scale))
    return (
        build_state(
            atlas.manifest, load_structures(atlas), sources, image_range, references
        ),
        mounts,
    )


class AtlasHandler(SimpleHTTPRequestHandler):
    state: dict = {}
    store = Path()
    mounts: dict[str, tuple[Path, str]] = {}

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def translate_path(self, path: str) -> str:
        request = path.partition("?")[0].partition("#")[0]
        for name, (source, _) in self.mounts.items():
            prefix = f"/ng/{name}"
            if request == prefix or request.startswith(prefix + "/"):
                suffix = request.removeprefix(prefix).lstrip("/")
                request = "/" + (source.relative_to(self.store) / suffix).as_posix()
                break
        return super().translate_path(request)

    def send_json(self, value: dict) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_range(self, request: str) -> bool:
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        path = Path(self.translate_path(request))
        if match is None or not path.is_file():
            return False
        size = path.stat().st_size
        start = int(match.group(1))
        end = min(int(match.group(2) or size - 1), size - 1)
        if start >= size or end < start:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return True
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            self.wfile.write(stream.read(end - start + 1))
        return True

    def do_GET(self) -> None:
        request = self.path.partition("?")[0]
        if self.send_range(request):
            return
        if request == "/ng_state.json":
            self.send_json(self.state)
            return
        for name, (source, scale) in self.mounts.items():
            if request != f"/ng/{name}/zarr.json":
                continue
            metadata = read_json(source / "zarr.json")
            datasets = multiscale(metadata)["datasets"]
            multiscale(metadata)["datasets"] = [
                dataset for dataset in datasets if dataset["path"] == scale
            ]
            self.send_json(metadata)
            return
        super().do_GET()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas")
    parser.add_argument(
        "--brainglobe-dir", type=Path, default=Path.home() / ".brainglobe"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--config", type=Path)
    output.add_argument("--s3-json", "--remote-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.s3_json:
        generate_s3_json(args.atlas, args.s3_json, args.brainglobe_dir)
        print(f"Wrote {args.s3_json}")
        return

    atlas = load_local_atlas(args.brainglobe_dir, args.atlas)
    if atlas is None:
        raise SystemExit(f"Atlas {args.atlas!r} is not installed")
    scale = ensure_local_data(atlas)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        partial(AtlasHandler, directory=str(atlas.store)),
    )
    host, port = server.server_address[:2]
    host = "localhost" if host in {"127.0.0.1", "0.0.0.0", "::"} else host
    origin = f"http://{host}:{port}"
    AtlasHandler.store = atlas.store
    AtlasHandler.state, AtlasHandler.mounts = make_local_state(atlas, scale, origin)
    if args.config:
        args.config.write_text(json.dumps(AtlasHandler.state, indent=2) + "\n")

    state_url = f"{origin}/ng_state.json"
    print(f"Open: https://neuroglancer-demo.appspot.com/#!{state_url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
