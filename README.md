# BrainGlobe local Neuroglancer viewer

Serve a BrainGlobe v3 atlas from the local Atlas API store to Neuroglancer:

```bash
python bg_ng_viewer.py allen_mouse_25um
```

For an official atlas, startup checks BrainGlobe S3 and downloads any missing
viewer components. Existing files with the expected size are reused.

If the atlas manifest does not exist in BrainGlobe S3, it is treated as a
local-only custom atlas. Required local components are validated and no remote
download is attempted.

The viewer exposes only the pyramid level matching the atlas resolution (for
example `s1` at 25 µm). This prevents Neuroglancer requesting higher-resolution
levels that BrainGlobe has not downloaded.

The canonical BrainGlobe v3 names are used directly: `template.ome.zarr`,
`annotations_compressed.ome.zarr`, `hemispheres.ome.zarr`,
`annotations.precomputed`, `terminology.csv`, and `manifest.json`.
