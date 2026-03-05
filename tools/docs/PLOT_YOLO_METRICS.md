# Plot YOLO Metrics

Script: `tools/plot_yolo_metrics.py`

Purpose:
- Build a grouped comparison bar chart from training `results.csv` files.
- Compare validation-like metrics stored inside YOLO training logs for several runs.

## Input

Reads:

```text
tools/runs/<run_name>/results.csv
```

Default metrics:
- `metrics/mAP50-95(B)`
- `metrics/mAP50(B)`
- `metrics/precision(B)`
- `metrics/recall(B)`
- `metrics/mAP50-95(M)`
- `metrics/mAP50(M)`
- `metrics/precision(M)`
- `metrics/recall(M)`

In the chart, `(M)` labels are displayed as `(P)` to match the segmentation-style naming used in discussion.

## Run

Recommended explicit run order:

```bash
C:\conda\python.exe tools/plot_yolo_metrics.py --runs yolo11n-seg_100_n yolo11s-seg_100_s yolo11m-seg_100_m yolo11l-seg_100_l yolo11x-seg_100_x
```

By default:
- `--epoch-policy best`
- best epoch is selected by `metrics/mAP50-95(M)`

Use last epoch instead:

```bash
C:\conda\python.exe tools/plot_yolo_metrics.py --runs yolo11n-seg_100_n yolo11s-seg_100_s yolo11m-seg_100_m yolo11l-seg_100_l yolo11x-seg_100_x --epoch-policy last
```

## Output

Files are written to:

```text
tools/runs/metric_plots/
```

Main outputs:
- `val_metrics_grouped_bar.png`
- `val_metrics_selected_epoch.csv`
- `run_versions.csv`

## Important

- This script uses metrics from training `results.csv`, not from a separate post-training `model.val()` pass.
- If you pass a wrong run name, the script now raises an error and lists available runs.
