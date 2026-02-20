# Training YOLO Segmentation (LV points)

Script: `tools/train_yolo_seg.py`

## 1) Install dependencies

```powershell
pip install ultralytics opencv-python
```

If `opencv-python` is problematic, you can use:

```powershell
pip install ultralytics pillow
```

## 1.1) Enable GPU in PyTorch (if `torch.cuda.is_available() == False`)

Check:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

If it shows CPU-only build, reinstall PyTorch with CUDA wheels (example CUDA 12.8):

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then verify again:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

## 2) Run training (YOLO11 medium, segmentation)

```powershell
python tools/train_yolo_seg.py `
  --input export_points_lv_800x600 `
  --model yolo11m-seg.pt `
  --epochs 100 `
  --max-gpu-util 95 `
  --max-vram-util 95 `
  --max-mem-ctrl-util 95 `
  --max-power-util 95 `
  --no-improve-minutes 15 `
  --batch 0.95 `
  --imgsz 800 `
  --device 0 `
  --strict-gpu-limits `
  --auto-reduce-batch `
  --run-name lv_seg `
  --new-run-folder
```

## 3) If GPU utilization is too high

Lower one or more parameters:

```powershell
--batch 0.90
--batch 0.85
--imgsz 640
--workers 2
```

If you want strict hard behavior (stop/retry when limits are exceeded), keep:

```powershell
--strict-gpu-limits --auto-reduce-batch --max-retries 4
```

## 4) Notes

- Mosaic is disabled in script (`mosaic=0.0`, `close_mosaic=0`).
- GPU metrics are printed during training: GPU util, VRAM%, memory controller util, power%, temperature.
- `--batch 0.95` uses Ultralytics AutoBatch targeting ~95% VRAM (single GPU).
- Early stop by time is enabled (default 15 minutes without improvement).
- New output folder is created every run by default.
- Runs are saved to `runs_yolo_seg/`.

## 5) YOLO12 (optional)

If your Ultralytics version supports YOLO12 segmentation weights, just change:

```powershell
--model yolo12m-seg.pt
```
