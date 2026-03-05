# Script Sequence

This file describes the practical script order that was built during the work on this repository.

## Main Pipeline

1. `tools/unify_datasets.py`
   - Optional preparation step if raw datasets need to be brought to a more uniform structure first.
   - Use this before export only if the raw dataset layout is still inconsistent.

2. `tools/export_points_annotations.py`
   - Reads raw datasets from `datasets/`.
   - Exports images and polygon point annotations into:

```text
export_points/
```

3. `tools/filter_left_ventricle_points.py`
   - Takes `export_points/`.
   - Keeps only left ventricle annotations.
   - Produces:

```text
export_points_lv/
```

4. `tools/resize_exported_points.py`
   - Takes `export_points_lv/`.
   - Normalizes orientation by dataset.
   - Resizes images to `800x600`.
   - Recomputes polygon coordinates.
   - In the current workflow also writes:
     - class id `0`
     - normalized coordinates `x / width`, `y / height`
   - Produces:

```text
export_points_lv_800x600/
```

5. `tools/yolo_learning.py`
   - Trains YOLO segmentation on:

```text
export_points_lv_800x600/dataset.yaml
```

   - Writes training runs into:

```text
tools/runs/
```

## Inference And Validation

6. `tools/yolo_single_test.py`
   - Runs inference on one image using one trained run from `tools/runs/`.
   - Saves preview images into:

```text
tools/runs/inference_preview/
```

7. `tools/infer_videos_to_gif.py`
   - Runs segmentation on videos from `videos/`.
   - Saves rendered GIF outputs into:

```text
videos/gif_results/
```

8. `tools/validate_yolo_runs.py`
   - Runs fresh `model.val()` for trained runs.
   - Saves post-training validation results into:

```text
tools/runs/validation_runs/
```

   - Main summary file:

```text
tools/runs/validation_runs/validation_summary.csv
```

## Metrics And Charts

9. `tools/plot_yolo_metrics.py`
   - Builds charts from training `results.csv`.
   - Use this if you want metrics exactly as stored during training.

10. `tools/plot_validation_metrics.py`
   - Builds charts from:

```text
tools/runs/validation_runs/validation_summary.csv
```

   - Use this if you want charts from a separate validation pass done after training.

## Recommended Practical Order

If the dataset is already prepared:

1. `tools/filter_left_ventricle_points.py`
2. `tools/resize_exported_points.py`
3. `tools/yolo_learning.py`
4. `tools/yolo_single_test.py`
5. `tools/infer_videos_to_gif.py` (optional, for video outputs)
6. `tools/validate_yolo_runs.py`
7. `tools/plot_validation_metrics.py`

If starting from raw source datasets:

1. `tools/unify_datasets.py` (only if needed)
2. `tools/export_points_annotations.py`
3. `tools/filter_left_ventricle_points.py`
4. `tools/resize_exported_points.py`
5. `tools/yolo_learning.py`
6. `tools/yolo_single_test.py`
7. `tools/infer_videos_to_gif.py` (optional, for video outputs)
8. `tools/validate_yolo_runs.py`
9. `tools/plot_yolo_metrics.py`
10. `tools/plot_validation_metrics.py`

## Notes

- `plot_yolo_metrics.py` and `plot_validation_metrics.py` are not interchangeable.
- `yolo_single_test.py` is for spot-check inference.
- `validate_yolo_runs.py` is the script that produces the new post-training metrics.
- `infer_videos_to_gif.py` is for batch video inference and GIF export.
