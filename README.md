# BrainGlobe Neuroglancer viewer

Generate a Neuroglancer state that streams an atlas from BrainGlobe S3:

```bash
python bg_ng_viewer.py allen_mouse_25um --s3-json allen_mouse_25um.json
```

The state points to each OME-Zarr group root, so Neuroglancer discovers its
resolution levels automatically. Colours, region labels, meshes, annotations,
and the hemisphere layer are included. `--remote-json` is an alias.
For batches, import `generate_s3_json` instead of starting subprocesses.

To serve the atlas from the local BrainGlobe store instead:

```bash
python bg_ng_viewer.py allen_mouse_25um
```
