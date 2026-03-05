# Export Images + Point Annotations

Script: `tools/export_points_annotations.py`

Exports data from:
- `datasets/EchoNet-Dynamic`
- `datasets/CAMUS_public`
- `datasets/cardiacUDC_dataset`

into PNG images and TXT polygon annotations.

## Annotation TXT Format

One polygon per line:

`class_id x1 y1 x2 y2 x3 y3 ...`

Coordinates are pixel coordinates.

## Output Structure

```text
export_points/
  echonet/
    images/
    labels/
  camus/
    images/
    labels/
  cardiac_udc/
    images/
    labels/
  manifest.csv
```

## Usage

Run all datasets:

```bash
python tools/export_points_annotations.py --root . --output export_points
```

Only one dataset:

```bash
python tools/export_points_annotations.py --root . --output export_points --datasets camus
```

Overwrite existing files:

```bash
python tools/export_points_annotations.py --root . --output export_points --overwrite
```

Quick test (limit per dataset):

```bash
python tools/export_points_annotations.py --root . --output export_points --limit 20
```
