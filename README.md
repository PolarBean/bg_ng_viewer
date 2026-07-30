# BrainGlobe Neuroglancer viewer

Generate a Neuroglancer state that streams an atlas from BrainGlobe S3:

```bash
python bg_ng_viewer.py allen_mouse_25um --s3-json allen_mouse_25um.json
```

To serve the atlas from the local BrainGlobe store instead:

```bash
python bg_ng_viewer.py allen_mouse_25um
```
