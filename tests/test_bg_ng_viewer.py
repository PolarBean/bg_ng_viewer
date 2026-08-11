import json
from pathlib import Path

import numpy as np
import pytest
import zstandard

import bg_ng_viewer as viewer


def manifest():
    return {
        "shape": [10, 20, 30],
        "resolution": [25.0, 20.0, 10.0],
        "template": {"location": "/templates/example/1_0"},
        "annotation_set": {"location": "/annotation-sets/example/3_0"},
        "terminology": {"location": "/terminologies/example/3_0"},
        "coordinate_space": {"location": "/coordinate-spaces/example/1_0"},
    }


def structures():
    return [
        {"annotation_value": "1", "color_hex_triplet": "#112233"},
        {"annotation_value": "2", "color_hex_triplet": "#abcdef"},
        {"annotation_value": "3", "color_hex_triplet": "#fedcba"},
    ]


def hierarchy(counts):
    """Structures with `counts[depth]` regions at each depth of the hierarchy."""
    rows, value = [], 0
    for depth, count in enumerate(counts):
        for _ in range(count):
            value += 1
            rows.append(
                {
                    "annotation_value": str(value),
                    "color_hex_triplet": "#000000",
                    "root_identifier_path": json.dumps(list(range(depth + 1))),
                }
            )
    return rows


def test_default_segments_fills_the_hierarchy_top_down():
    rows = hierarchy([1, 10, 50, 200])
    chosen = viewer.default_segments(rows, limit=40)

    depths = {int(row["annotation_value"]): viewer.structure_depth(row) for row in rows}
    by_depth = [
        sum(1 for value in chosen if depths[value] == depth) for depth in range(4)
    ]
    assert len(chosen) == 40
    assert by_depth == [1, 10, 29, 0]
    assert viewer.default_segments(rows, limit=40) == chosen


def test_default_segments_keeps_everything_in_a_small_atlas():
    rows = hierarchy([1, 3, 5])
    assert len(viewer.default_segments(rows, limit=40)) == 9


def test_atlas_builds_component_paths_from_manifest():
    atlas = viewer.Atlas(manifest())

    assert atlas.relative("template", viewer.TEMPLATE) == (
        "templates/example/1_0/template.ome.zarr"
    )
    assert atlas.relative("annotation_set", viewer.PRECOMPUTED) == (
        "annotation-sets/example/3_0/annotations.precomputed"
    )
    assert viewer.local_template_range(atlas) is None


def test_ensure_local_data_does_not_contact_s3_for_complete_api_install(
    monkeypatch,
):
    atlas = object()
    monkeypatch.setattr(viewer, "validate_local_atlas", lambda candidate: "s0")

    # Local serving must never construct a remote filesystem.
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "s3fs":
            raise AssertionError("complete local atlases must not contact S3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    assert viewer.ensure_local_data(atlas) == "s0"


def test_build_state_preserves_layers_colours_and_meshes():
    sources = {
        "template": "template/|zarr3:",
        "annotation": "annotation/|zarr3:",
        "hemisphere": "hemisphere/|zarr3:",
        "precomputed": "precomputed://meshes",
    }
    state = viewer.build_state(manifest(), structures(), sources, (2.0, 42.0))

    template, annotation, meshes, hemisphere = state["layers"]
    assert template["shaderControls"]["normalized"]["range"] == [2.0, 42.0]
    assert template["volumeRendering"] == "off"
    # Every region is painted in the cross sections, without a mesh subsource.
    assert annotation["segments"] == ["1", "2", "3"]
    assert annotation["segmentColors"]["2"] == "#abcdef"
    assert annotation["source"][1]["url"] == "precomputed://meshes"
    assert annotation["source"][1]["subsources"] == {"properties": True}
    # Meshes live in their own, volume-less layer, so they only show in 3D.
    assert meshes["name"] == "region meshes"
    assert meshes["segments"] == ["1", "2", "3"]
    assert meshes["source"]["subsources"] == {"properties": True, "mesh": True}
    assert hemisphere["visible"] is False
    assert [state["dimensions"][axis][0] for axis in "xyz"] == pytest.approx(
        [10e-6, 20e-6, 25e-6]
    )
    assert state["position"] == [15.0, 10.0, 5.0]


def test_build_state_shows_all_regions_in_2d_and_40_meshes_in_3d():
    sources = {
        "template": "template/|zarr3:",
        "annotation": "annotation/|zarr3:",
        "hemisphere": "hemisphere/|zarr3:",
        "precomputed": "precomputed://meshes",
    }
    rows = hierarchy([1, 10, 50, 200])
    _, annotation, meshes, _ = viewer.build_state(
        manifest(), rows, sources, (2.0, 42.0)
    )["layers"]

    assert not any(segment.startswith("!") for segment in annotation["segments"])
    assert len(annotation["segments"]) == len(rows)
    assert sum(not s.startswith("!") for s in meshes["segments"]) == 40
    assert len(meshes["segments"]) == len(rows)


def test_build_state_adds_additional_references_as_hidden_layers():
    with_references = manifest() | {
        "additional_references": [
            {"name": "example-nissl-template", "location": "/templates/nissl/3_0"}
        ]
    }
    sources = {
        "template": "template/|zarr3:",
        "annotation": "annotation/|zarr3:",
        "hemisphere": "hemisphere/|zarr3:",
    }
    state = viewer.build_state(
        with_references,
        structures(),
        sources,
        (2.0, 42.0),
        [("example-nissl", "nissl/|zarr3:", (1.0, 7.0))],
    )

    names = [layer["name"] for layer in state["layers"]]
    assert names == ["template", "example-nissl", "annotation", "hemisphere mask"]
    nissl = state["layers"][1]
    assert nissl["type"] == "image"
    assert nissl["visible"] is False
    assert nissl["shaderControls"]["normalized"]["range"] == [1.0, 7.0]


def test_generate_s3_json_samples_each_additional_reference(
    tmp_path: Path, monkeypatch
):
    store = tmp_path / viewer.STORE_NAME
    atlas_dir = store / "atlases/example_25um/3_0"
    terminology_dir = store / "terminologies/example/3_0"
    atlas_dir.mkdir(parents=True)
    terminology_dir.mkdir(parents=True)
    with_references = manifest() | {
        "additional_references": [
            {"name": "example-nissl-template", "location": "/templates/nissl/3_0"}
        ]
    }
    (atlas_dir / viewer.MANIFEST).write_text(json.dumps(with_references))
    (terminology_dir / viewer.TERMINOLOGY).write_text(
        "annotation_value,color_hex_triplet\n1,#112233\n"
    )
    local_atlas = viewer.Atlas(
        with_references, store, atlas_dir / viewer.MANIFEST
    )
    monkeypatch.setattr(viewer, "load_local_atlas", lambda *_: local_atlas)
    sampled = []

    def fake_range(url):
        sampled.append(url)
        return (0.0, 1.0)

    monkeypatch.setattr(viewer, "remote_template_range", fake_range)

    state = viewer.generate_s3_state("example_25um", tmp_path)
    reference = state["layers"][1]

    assert reference["name"] == "example-nissl"
    assert reference["source"] == (
        f"{viewer.HTTP_ROOT}/templates/nissl/3_0/template.ome.zarr/|zarr3:"
    )
    assert sampled == [
        f"{viewer.HTTP_ROOT}/templates/example/1_0/template.ome.zarr",
        f"{viewer.HTTP_ROOT}/templates/nissl/3_0/template.ome.zarr",
    ]


def test_sampled_zarr_range_calculates_minimum_and_maximum():
    data = np.arange(1000, dtype="<u2").reshape(10, 10, 10)
    metadata = {
        "shape": list(data.shape),
        "data_type": "uint16",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(data.shape)},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0,
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {}},
        ],
    }
    encoded = zstandard.ZstdCompressor().compress(data.tobytes())

    assert viewer.sampled_zarr_range(metadata, lambda _: encoded) == [0.0, 999.0]


def test_generate_s3_json_uses_local_metadata(tmp_path: Path, monkeypatch):
    store = tmp_path / viewer.STORE_NAME
    atlas_dir = store / "atlases/example_25um/3_0"
    terminology_dir = store / "terminologies/example/3_0"
    atlas_dir.mkdir(parents=True)
    terminology_dir.mkdir(parents=True)
    (atlas_dir / viewer.MANIFEST).write_text(json.dumps(manifest()))
    (terminology_dir / viewer.TERMINOLOGY).write_text(
        "annotation_value,color_hex_triplet\n" "1,#112233\n" "2,#abcdef\n" "3,#fedcba\n"
    )
    local_atlas = viewer.Atlas(manifest(), store, atlas_dir / viewer.MANIFEST)
    monkeypatch.setattr(viewer, "load_local_atlas", lambda *_: local_atlas)
    monkeypatch.setattr(viewer, "remote_template_range", lambda _: (4.0, 96.0))

    output = viewer.generate_s3_json("example_25um", tmp_path / "state.json", tmp_path)
    state = json.loads(output.read_text())

    assert output == tmp_path / "state.json"
    assert state["layers"][0]["source"].startswith(viewer.HTTP_ROOT)
    assert state["layers"][0]["shaderControls"]["normalized"]["range"] == [
        4.0,
        96.0,
    ]
    assert state["layers"][1]["segmentColors"]["2"] == "#abcdef"


def test_translate_path_matches_url_encoded_mount_name(tmp_path: Path):
    store = tmp_path / viewer.STORE_NAME
    source = store / "templates/ccfv2_mouse/1_0/template.ome.zarr"
    handler = object.__new__(viewer.AtlasHandler)
    handler.store = store
    handler.directory = str(store)
    handler.mounts = {"ccfv2_mouse-Averaged reference": (source, "s0")}

    translated = handler.translate_path(
        "/ng/ccfv2_mouse-Averaged%20reference/zarr.json"
    )

    assert translated == str(source / "zarr.json")
