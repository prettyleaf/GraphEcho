# Validate YOLO Runs

Script: `tools/validate_yolo_runs.py`

Purpose:
- Run `model.val()` for trained YOLO segmentation models.
- Save fresh validation metrics into a single summary CSV.
- Avoid the Windows `multiprocessing` issue that happened when validation was started from `python -`.

## Why This Script Exists

Running validation through stdin like:

```bash
@' ... '@ | C:\conda\python.exe -
```

can fail on Windows because `Ultralytics` may spawn worker processes and child processes cannot reopen `<stdin>` as a normal Python file.

This script fixes that by:
- running from a normal `.py` file
- using `workers=0` by default

## Default Dataset

Default dataset config:

```text
export_points_lv_800x600/dataset.yaml
```

This validates the standard `val.txt` split.

## Run

Validate selected runs on the normal validation split:

```bash
C:\conda\python.exe tools/validate_yolo_runs.py --runs yolo11n-seg_100_n yolo11s-seg_100_s yolo11m-seg_100_m yolo11l-seg_100_l yolo11x-seg_100_x --batch 8 --workers 0 --device 0
```

Validate on all labeled images instead of only `val.txt`:

```bash
C:\conda\python.exe tools/validate_yolo_runs.py --runs yolo11n-seg_100_n yolo11s-seg_100_s yolo11m-seg_100_m yolo11l-seg_100_l yolo11x-seg_100_x --all-images --batch 8 --workers 0 --device 0
```

Validate a single run:

```bash
C:\conda\python.exe tools/validate_yolo_runs.py --runs yolo11x-seg_100_x --batch 8 --workers 0 --device 0
```

## Output

Validation artifacts are written to:

```text
tools/runs/validation_runs/
```

Main summary file:

```text
tools/runs/validation_runs/validation_summary.csv
```

Per-run validation folders:

```text
tools/runs/validation_runs/<run_name>_val/
```

## Metrics Saved

- `metrics/mAP50-95(B)`
- `metrics/mAP50(B)`
- `metrics/precision(B)`
- `metrics/recall(B)`
- `metrics/mAP50-95(M)`
- `metrics/mAP50(M)`
- `metrics/precision(M)`
- `metrics/recall(M)`

## Temporary Files For `--all-images`

When `--all-images` is used, the script generates:

```text
export_points_lv_800x600/_all_images.txt
export_points_lv_800x600/_all_images.yaml
```

These are used to validate against the full labeled image set.
