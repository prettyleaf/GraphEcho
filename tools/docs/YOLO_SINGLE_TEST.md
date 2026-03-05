# YOLO Single Test

Script: `tools/yolo_single_test.py`

Purpose:
- Run segmentation inference on a single image using one of the trained runs in `tools/runs/`.
- Save the rendered prediction preview into `tools/runs/inference_preview/`.

## What It Does

- Looks for trained runs in:

```text
tools/runs/*/weights/best.pt
```

- Supports:
  - explicit run selection with `--run`
  - automatic best-run selection with `--run auto`
- Loads an image from:
  - explicit `--image`
  - or first available image from `export_points_lv_800x600/*/images/*.png`
- Normalizes image to `uint8`
- Converts grayscale input to BGR if needed
- Runs `YOLO(...)(img)`
- Saves plotted prediction image

## Run

Auto-pick the best run by `metrics/mAP50-95(M)` from `results.csv`:

```bash
C:\conda\python.exe tools/yolo_single_test.py
```

Run a specific model:

```bash
C:\conda\python.exe tools/yolo_single_test.py --run yolo11l-seg_100_l
```

Run on a specific image:

```bash
C:\conda\python.exe tools/yolo_single_test.py --run yolo11x-seg_100_x --image export_points_lv_800x600/camus/images/database_nifti_patient0001_patient0001_2CH_ED.png
```

Show the prediction window:

```bash
C:\conda\python.exe tools/yolo_single_test.py --run yolo11m-seg_100_m --show
```

## Output

Prediction previews are saved to:

```text
tools/runs/inference_preview/
```

Example:

```text
tools/runs/inference_preview/yolo11l-seg_100_l_database_nifti_patient0001_patient0001_2CH_ED_pred.png
```

## Notes

- `auto` does not mean "largest model"; it means "best run by metric from training `results.csv`".
- If you need a specific model, always pass `--run`.
