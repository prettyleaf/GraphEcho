# YOLO Learning

Script: `tools/yolo_learning.py`

Purpose:
- Train a YOLO segmentation model on the prepared LV-only dataset.
- Uses the resized dataset in `export_points_lv_800x600/`.
- Current script version is fixed to one training configuration and does not use CLI arguments.

## Current Configuration

- Model: `yolo11l-seg.pt`
- Epochs: `100`
- Dataset: `../export_points_lv_800x600/dataset.yaml`
- Image size: `800`
- Batch: `20`
- Mosaic: `0.0`
- Output run name: `yolo11l-seg_100_l`

## Important

This script currently expects to be launched from the `tools/` directory because:
- model weights are referenced as `yolo11l-seg.pt`
- dataset path is referenced as `../export_points_lv_800x600/dataset.yaml`

If you run it from repository root without changing the script, paths will be wrong.

## Run

```bash
cd tools
C:\conda\python.exe yolo_learning.py
```

## Dataset Assumptions

- Dataset is LV-only.
- Class id is `0`.
- Labels are polygon segmentation labels.
- Images are already resized to `800x600`.

## Output

Training outputs are written into:

```text
tools/runs/<run_name>/
```

For the current configuration:

```text
tools/runs/yolo11l-seg_100_l/
```

## Notes

- `ClearML` integration is currently commented out.
- `device=[0, 1]` is hardcoded in the script.
- `exist_ok=True` means the run folder can be reused instead of forcing a new unique folder.
