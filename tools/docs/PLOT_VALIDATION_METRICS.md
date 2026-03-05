# Plot Validation Metrics

Script: `tools/plot_validation_metrics.py`

Purpose:
- Build a separate grouped bar chart from the fresh metrics produced by `tools/validate_yolo_runs.py`.
- This is the post-validation chart, not the chart from training `results.csv`.

## Input

Reads:

```text
tools/runs/validation_runs/validation_summary.csv
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

Displayed labels convert `(M)` to `(P)` for the chart.

## Run

Recommended run order:

```bash
C:\conda\python.exe tools/plot_validation_metrics.py --runs yolo11n-seg_100_n yolo11s-seg_100_s yolo11m-seg_100_m yolo11l-seg_100_l yolo11x-seg_100_x
```

## Output

Files are written to:

```text
tools/runs/validation_runs/
```

Main outputs:
- `validation_metrics_grouped_bar.png`
- `validation_metrics_selected.csv`
- `validation_run_versions.csv`

## Difference From `plot_yolo_metrics.py`

- `plot_yolo_metrics.py` uses training `results.csv`
- `plot_validation_metrics.py` uses fresh metrics from `model.val()`

Use this script when you want the chart to reflect an actual separate validation pass.
