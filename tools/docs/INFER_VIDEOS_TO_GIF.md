# Infer Videos To GIF

Script: `tools/infer_videos_to_gif.py`

Purpose:
- Run trained YOLO segmentation on all videos in a folder.
- Render predictions frame-by-frame.
- Export each processed video into a GIF file.

## Input

- Video folder (default):

```text
videos/
```

Supported extensions:
- `.avi`
- `.mp4`
- `.mov`
- `.mkv`
- `.wmv`
- `.mpeg`
- `.mpg`

## Model Selection

- `--run auto`:
  - first tries best run from `tools/runs/validation_runs/validation_summary.csv` by `metrics/mAP50-95(M)`
  - fallback: best run by `metrics/mAP50-95(M)` from training `results.csv`
- or pass an explicit run name:

```text
yolo11x-seg_100_x
```

## Run

Default workflow:

```bash
C:\conda\python.exe tools/infer_videos_to_gif.py --input-dir videos --output-dir videos/gif_results --run auto
```

Use a specific model:

```bash
C:\conda\python.exe tools/infer_videos_to_gif.py --input-dir videos --output-dir videos/gif_results --run yolo11x-seg_100_x
```

Lighter GIF files:

```bash
C:\conda\python.exe tools/infer_videos_to_gif.py --input-dir videos --output-dir videos/gif_results --run auto --frame-step 2 --gif-fps 8
```

Quick test:

```bash
C:\conda\python.exe tools/infer_videos_to_gif.py --input-dir videos --output-dir videos/gif_results_test --run auto --max-frames 30
```

## Output

Default output folder:

```text
videos/gif_results/
```

Output naming pattern:

```text
<video_stem>_<run_name>.gif
```

Example:

```text
videos/gif_results/30200820250813_Nikiforov_20070926_20250813082410459_yolo11x-seg_100_x.gif
```

## Parameters

- `--conf`: confidence threshold (default `0.25`)
- `--frame-step`: process every Nth frame (default `1`)
- `--gif-fps`: output GIF FPS (default `12`)
- `--max-frames`: max frames per video, `0` means all frames

## Notes

- Script reads models from `tools/runs/*/weights/best.pt`.
- If no videos are found, script raises an explicit error.
- GIFs can be large for long videos with `frame-step=1`.
