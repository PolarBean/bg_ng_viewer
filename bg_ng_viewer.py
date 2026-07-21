#!/usr/bin/env python3
"""Serve a BrainGlobe v3 atlas to Neuroglancer."""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ATLAS_STORE = "brainglobe-atlasapi"
TEMPLATE_NAME = "template.ome.zarr"
ANNOTATION_NAME = "annotations_compressed.ome.zarr"
HEMISPHERE_NAME = "hemispheres.ome.zarr"
PRECOMPUTED_NAME = "annotations.precomputed"
TERMINOLOGY_NAME = "terminology.csv"
MANIFEST_NAME = "manifest.json"
REMOTE_ROOT = "brainglobe/atlas-rc2"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def version_key(path: Path) -> tuple[int, ...]:
    return tuple(int(part) for part in path.parent.name.replace(".", "_").split("_"))


def atlas_store(root: Path) -> Path:
    root = root.expanduser().resolve()
    return root if (root / "atlases").is_dir() else root / ATLAS_STORE


def find_manifest(store: Path, atlas_name: str) -> Path:
    manifests = list((store / "atlases" / atlas_name).glob(f"*/{MANIFEST_NAME}"))
    if manifests:
        return max(manifests, key=version_key)

    installed = sorted(path.parent.parent.name for path in (store / "atlases").glob(f"*/*/{MANIFEST_NAME}"))
    names = ", ".join(installed) or "none"
    raise SystemExit(f"Atlas {atlas_name!r} is not installed in {store}. Installed: {names}")


@dataclass(frozen=True)
class AtlasPaths:
    store: Path
    manifest_file: Path
    manifest: dict
    template: Path
    annotation: Path
    hemisphere: Path
    terminology: Path
    coordinate_manifest: Path
    precomputed: Path

    @classmethod
    def load(cls, root: Path, atlas_name: str) -> "AtlasPaths":
        store = atlas_store(root)
        manifest_file = find_manifest(store, atlas_name)
        manifest = read_json(manifest_file)

        def location(component: str) -> Path:
            return store / manifest[component]["location"].lstrip("/")

        template_dir = location("template")
        annotation_dir = location("annotation_set")
        return cls(
            store=store,
            manifest_file=manifest_file,
            manifest=manifest,
            template=template_dir / TEMPLATE_NAME,
            annotation=annotation_dir / ANNOTATION_NAME,
            hemisphere=annotation_dir / HEMISPHERE_NAME,
            terminology=location("terminology") / TERMINOLOGY_NAME,
            coordinate_manifest=location("coordinate_space") / MANIFEST_NAME,
            precomputed=annotation_dir / PRECOMPUTED_NAME,
        )

    def remote(self, path: Path) -> str:
        return f"{REMOTE_ROOT}/{path.relative_to(self.store).as_posix()}"


def atlas_scale(zarr_path: Path, resolution_um: float) -> str:
    multiscale = read_json(zarr_path / "zarr.json")["attributes"]["ome"]["multiscales"][0]
    for dataset in multiscale["datasets"]:
        scale_mm = dataset["coordinateTransformations"][0]["scale"][0]
        if abs(scale_mm * 1000 - resolution_um) < 1e-6:
            return dataset["path"]
    raise SystemExit(f"No {resolution_um:g} µm scale in {zarr_path}")


def require_chunks(zarr_path: Path, scale: str, label: str) -> None:
    chunk_dir = zarr_path / scale / "c"
    if chunk_dir.is_dir() and any(path.is_file() for path in chunk_dir.rglob("*")):
        return
    raise SystemExit(f"{label} chunks are missing from {chunk_dir}")


def sync_file(fs, remote: str, local: Path, label: str) -> None:
    size = fs.info(remote)["size"]
    if local.is_file() and local.stat().st_size == size:
        print(f"Checked {label}: complete")
        return
    print(f"Downloading {label}...")
    local.parent.mkdir(parents=True, exist_ok=True)
    fs.get(remote, local)


def sync_tree(fs, remote: str, local: Path, label: str) -> None:
    downloads = []
    for remote_path, details in fs.find(remote, detail=True).items():
        if details.get("type") == "directory":
            continue
        destination = local / Path(remote_path).relative_to(remote)
        if not destination.is_file() or destination.stat().st_size != details["size"]:
            downloads.append((remote_path, destination))

    if not downloads:
        print(f"Checked {label}: complete")
        return

    print(f"Downloading {len(downloads)} missing {label} files...")

    def download(item) -> None:
        remote_path, destination = item
        destination.parent.mkdir(parents=True, exist_ok=True)
        fs.get(remote_path, destination)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(download, downloads))
    print(f"Downloaded {label}")


def create_hemisphere_mask(paths: AtlasPaths, scale: str) -> None:
    if paths.hemisphere.is_dir():
        return

    import numpy as np
    import zarr

    print("Creating symmetric hemisphere OME-Zarr...")
    attributes = read_json(paths.annotation / "zarr.json")["attributes"]
    multiscale = attributes["ome"]["multiscales"][0]
    multiscale["datasets"] = [dataset for dataset in multiscale["datasets"] if dataset["path"] == scale]

    source = zarr.open_array(paths.annotation / scale, mode="r")
    root = zarr.open_group(paths.hemisphere, mode="w", attributes=attributes)
    target = root.create_array(
        scale,
        shape=source.shape,
        chunks=source.chunks,
        dtype="uint8",
        dimension_names=source.metadata.dimension_names,
    )
    middle = source.shape[-1] // 2
    target[..., :middle] = np.uint8(2)
    target[..., middle:] = np.uint8(1)
    print("Created hemisphere mask")


def validate_local_atlas(paths: AtlasPaths) -> str:
    required = (
        paths.template / "zarr.json",
        paths.annotation / "zarr.json",
        paths.terminology,
        paths.coordinate_manifest,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Custom atlas is missing required files:\n" + "\n".join(map(str, missing)))

    scale = atlas_scale(paths.template, float(paths.manifest["resolution"][0]))
    require_chunks(paths.template, scale, "Template")
    require_chunks(paths.annotation, scale, "Annotation")

    if paths.manifest.get("symmetric", False):
        create_hemisphere_mask(paths, scale)
    elif not paths.hemisphere.is_dir():
        raise SystemExit(f"Custom asymmetric atlas is missing {paths.hemisphere}")

    print("Validated local custom atlas")
    return scale


def ensure_components(paths: AtlasPaths) -> str:
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    remote_manifest = paths.remote(paths.manifest_file)
    try:
        official_atlas = fs.exists(remote_manifest)
    except OSError:
        official_atlas = False

    if not official_atlas:
        print("No BrainGlobe S3 package found; treating atlas as local-only")
        return validate_local_atlas(paths)

    sync_file(fs, paths.remote(paths.template / "zarr.json"), paths.template / "zarr.json", "template metadata")
    sync_file(fs, paths.remote(paths.annotation / "zarr.json"), paths.annotation / "zarr.json", "annotation metadata")
    sync_file(fs, paths.remote(paths.terminology), paths.terminology, "terminology")
    sync_file(fs, paths.remote(paths.coordinate_manifest), paths.coordinate_manifest, "coordinate-space metadata")

    scale = atlas_scale(paths.template, float(paths.manifest["resolution"][0]))
    sync_tree(fs, paths.remote(paths.template / scale), paths.template / scale, "template chunks")
    sync_tree(fs, paths.remote(paths.annotation / scale), paths.annotation / scale, "annotation chunks")
    sync_file(fs, paths.remote(paths.precomputed / "info"), paths.precomputed / "info", "precomputed info")
    sync_tree(fs, paths.remote(paths.precomputed / "mesh"), paths.precomputed / "mesh", "mesh and index")
    sync_tree(
        fs,
        paths.remote(paths.precomputed / "segment_properties"),
        paths.precomputed / "segment_properties",
        "segment properties",
    )

    if paths.manifest.get("symmetric", False):
        create_hemisphere_mask(paths, scale)
    else:
        sync_file(fs, paths.remote(paths.hemisphere / "zarr.json"), paths.hemisphere / "zarr.json", "hemisphere metadata")
        sync_tree(fs, paths.remote(paths.hemisphere / scale), paths.hemisphere / scale, "hemisphere chunks")
    return scale


def mesh_ids(precomputed: Path) -> set[str]:
    mesh_dir = precomputed / "mesh"
    if not mesh_dir.is_dir():
        return set()
    ids = {path.name for path in mesh_dir.iterdir() if path.is_file() and path.name != "info" and not path.name.endswith(".index")}
    missing_indexes = [identifier for identifier in ids if not (mesh_dir / f"{identifier}.index").is_file()]
    if missing_indexes:
        raise SystemExit(f"{len(missing_indexes)} mesh index files are missing from {mesh_dir}")
    for metadata in (precomputed / "info", mesh_dir / "info"):
        if not metadata.is_file():
            raise SystemExit(f"Missing precomputed metadata: {metadata}")
    return ids


def template_range(template: Path, scale: str) -> list[float]:
    import numpy as np
    import zarr

    array = zarr.open_array(template / scale, mode="r")
    step = tuple(max(1, length // 64) for length in array.shape)
    sample = np.asarray(array[tuple(slice(None, None, value) for value in step)])
    low, high = np.percentile(sample, (1, 99.5))
    if high <= low:
        low, high = sample.min(), sample.max()
    return [float(low), float(high)]


def make_state(paths: AtlasPaths, scale: str, origin: str) -> tuple[dict, dict[str, Path]]:
    with paths.terminology.open(newline="") as stream:
        structures = list(csv.DictReader(stream))

    ids = [str(int(row["annotation_value"])) for row in structures]
    colors = {str(int(row["annotation_value"])): row["color_hex_triplet"] for row in structures}
    source = lambda name: f"{origin}/ng/{name}/|zarr3:"
    mounts = {"template": paths.template, "annotation": paths.annotation, "hemisphere": paths.hemisphere}
    layers = [
        {
            "type": "image",
            "source": source("template"),
            "shaderControls": {"normalized": {"range": template_range(paths.template, scale)}},
            "name": "template",
        },
        {
            "type": "segmentation",
            "source": source("annotation"),
            "segments": ids,
            "segmentColors": colors,
            "objectAlpha": 0.55,
            "name": "annotation",
        },
        {
            "type": "segmentation",
            "source": source("hemisphere"),
            "segments": ["1", "2"],
            "segmentColors": {"1": "#4c78a8", "2": "#f58518"},
            "visible": False,
            "name": "hemisphere mask",
        },
    ]

    available_meshes = mesh_ids(paths.precomputed)
    if available_meshes:
        layers[1]["segments"] = sorted(set(ids) & available_meshes, key=int)
        layers[1]["source"] = [
            layers[1]["source"],
            {
                "url": f"precomputed://{origin}/{paths.precomputed.relative_to(paths.store).as_posix()}",
                "enableDefaultSubsources": False,
                "subsources": {"properties": True, "mesh": True},
            },
        ]

    shape = list(reversed(paths.manifest["shape"]))
    resolution = list(reversed(paths.manifest["resolution"]))
    state = {
        "dimensions": {axis: [value * 1e-6, "m"] for axis, value in zip("xyz", resolution)},
        "position": [value / 2 for value in shape],
        "crossSectionScale": 1,
        "projectionScale": max(shape) * 1.25,
        "layers": layers,
        "showSlices": False,
        "selectedLayer": {"visible": True, "layer": "annotation"},
        "layout": "4panel-alt",
    }
    return state, mounts


class AtlasHandler(SimpleHTTPRequestHandler):
    state: dict = {}
    store = Path()
    mounts: dict[str, Path] = {}
    scale = ""

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        request_path = self.path.partition("?")[0]
        if request_path == "/ng_state.json" or request_path.endswith("zarr.json"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def translate_path(self, path: str) -> str:
        request_path = path.partition("?")[0].partition("#")[0]
        for name, source in self.mounts.items():
            prefix = f"/ng/{name}"
            if request_path == prefix or request_path.startswith(prefix + "/"):
                suffix = request_path.removeprefix(prefix).lstrip("/")
                local_path = source.relative_to(self.store) / suffix
                return super().translate_path("/" + local_path.as_posix())
        return super().translate_path(path)

    def send_json(self, value: dict) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_range(self, request_path: str) -> bool:
        header = self.headers.get("Range")
        if header is None:
            return False
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", header)
        local_path = Path(self.translate_path(request_path))
        if match is None or not local_path.is_file():
            return False

        size = local_path.stat().st_size
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else size - 1
        if start >= size or end < start:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return True

        end = min(end, size - 1)
        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(local_path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with local_path.open("rb") as stream:
            stream.seek(start)
            self.wfile.write(stream.read(length))
        return True

    def do_GET(self) -> None:
        request_path = self.path.partition("?")[0]
        if self.send_range(request_path):
            return
        if request_path == "/ng_state.json":
            self.send_json(self.state)
            return

        for name, source in self.mounts.items():
            direct_path = "/" + (source / "zarr.json").relative_to(self.store).as_posix()
            if request_path not in {f"/ng/{name}/zarr.json", direct_path}:
                continue
            metadata = read_json(source / "zarr.json")
            multiscale = metadata["attributes"]["ome"]["multiscales"][0]
            multiscale["datasets"] = [dataset for dataset in multiscale["datasets"] if dataset["path"] == self.scale]
            self.send_json(metadata)
            return

        super().do_GET()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", help="installed BrainGlobe v3 atlas name")
    parser.add_argument("--brainglobe-dir", type=Path, default=Path.home() / ".brainglobe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--config", type=Path, help="write the generated Neuroglancer state")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = AtlasPaths.load(args.brainglobe_dir, args.atlas)
    scale = ensure_components(paths)

    handler = partial(AtlasHandler, directory=str(paths.store))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    host, port = server.server_address[:2]
    public_host = "localhost" if host in {"127.0.0.1", "0.0.0.0", "::"} else host
    origin = f"http://{public_host}:{port}"

    AtlasHandler.store = paths.store
    AtlasHandler.scale = scale
    AtlasHandler.state, AtlasHandler.mounts = make_state(paths, scale, origin)

    if args.config:
        args.config.write_text(json.dumps(AtlasHandler.state, indent=2) + "\n")

    state_url = f"{origin}/ng_state.json"
    print(f"Mounted: {paths.store}")
    print(f"Files:   {origin}/")
    print(f"State:   {state_url}")
    print(f"Open:    https://neuroglancer-demo.appspot.com/#!{state_url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
