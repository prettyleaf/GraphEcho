# Filter Left Ventricle Only

Script: `tools/filter_left_ventricle_points.py`

Purpose:
- Read exported points dataset (`export_points/`).
- Keep only polygons with LV class id (`1` by default).
- Keep only samples where LV polygons exist.
- Write into a separate folder (e.g. `export_points_lv/`).

## Run

```bash
python tools/filter_left_ventricle_points.py --input export_points --output export_points_lv --class-id 1 --clean-output --overwrite
```

Then run resize+orientation pipeline on this LV-only folder:

```bash
python tools/resize_exported_points.py --input export_points_lv --output export_points_lv_800x600 --width 800 --height 600 --mode stretch --overwrite
```
