# Resize Exported Images To 800x600 + Rebuild Point TXT

Script: `tools/resize_exported_points.py`

## What it does

1. Reads exported datasets from `export_points/` (or custom input).
2. Normalizes orientation before resize:
   - `cardiac_udc`: rotate 90° clockwise (left -> up)
   - `camus`: rotate 90° counter-clockwise (right -> up)
   - `echonet`: rotate 180° (down -> up)
3. Resizes every image to target size (default `800x600`).
4. Recomputes all polygon point coordinates in `.txt`.
4. Clips points to image bounds `[0..W-1], [0..H-1]`.
5. Writes result into a new folder and generates:
   - `manifest.csv`
   - `summary.json`

## Annotation format

Each line:

`class_id x1 y1 x2 y2 x3 y3 ...`

## Run

```bash
python tools/resize_exported_points.py --input export_points --output export_points_800x600 --width 800 --height 600 --mode stretch --overwrite
```

Disable orientation normalization (if needed):

```bash
python tools/resize_exported_points.py --input export_points --output export_points_800x600 --disable-orientation
```

### Modes

- `stretch` (default): exact 800x600, anisotropic scaling.
- `fit`: keep aspect ratio + black padding to 800x600.

## Quick test

```bash
python tools/resize_exported_points.py --input export_points --output export_points_800x600_test --limit 20 --overwrite
```
