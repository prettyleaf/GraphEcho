from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "tools" / "runs"
DATASET_YAML = ROOT / "export_points_lv_800x600" / "dataset.yaml"
OUT_DIR = RUNS_DIR / "validation_runs"
SUMMARY_CSV = OUT_DIR / "validation_summary.csv"
TMP_ALL_LIST = ROOT / "export_points_lv_800x600" / "_all_images.txt"
TMP_ALL_YAML = ROOT / "export_points_lv_800x600" / "_all_images.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO segmentation runs on the full dataset.")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help="Specific run folder names inside tools/runs. Default: auto-discover all runs with weights/best.pt.",
    )
    parser.add_argument("--data", type=Path, default=DATASET_YAML, help="Path to dataset.yaml")
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Validate on all labeled images under export_points_lv_800x600 instead of dataset val split.",
    )
    parser.add_argument("--imgsz", type=int, default=800, help="Validation image size")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0 or cpu")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers. 0 avoids Windows spawn issues.")
    parser.add_argument("--split", default="val", help="Dataset split to validate")
    parser.add_argument("--plots", action="store_true", help="Save validation plots")
    return parser.parse_args()


def discover_run_dirs(runs_dir: Path, selected: List[str]) -> List[Path]:
    if selected:
        missing = []
        out = []
        for name in selected:
            run_dir = runs_dir / name
            best_pt = run_dir / "weights" / "best.pt"
            if not best_pt.exists():
                missing.append(name)
                continue
            out.append(run_dir)
        if missing:
            available = sorted(
                d.name for d in runs_dir.iterdir() if d.is_dir() and (d / "weights" / "best.pt").exists()
            )
            raise FileNotFoundError(
                "Requested run(s) not found: " + ", ".join(missing) + ". Available runs: " + ", ".join(available)
            )
        return out

    return sorted(
        d for d in runs_dir.iterdir() if d.is_dir() and (d / "weights" / "best.pt").exists()
    )


def safe_metric(metrics_obj, key: str) -> Optional[float]:
    value = metrics_obj.results_dict.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_all_images_yaml(dataset_root: Path) -> Path:
    image_paths = sorted(dataset_root.rglob("images/*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {dataset_root}")

    TMP_ALL_LIST.write_text(
        "\n".join(p.as_posix() for p in image_paths) + "\n",
        encoding="utf-8",
    )
    TMP_ALL_YAML.write_text(
        "path: " + dataset_root.as_posix() + "\n"
        "train: train.txt\n"
        "val: " + TMP_ALL_LIST.as_posix() + "\n"
        "names:\n"
        "  0: left_ventricle\n",
        encoding="utf-8",
    )
    return TMP_ALL_YAML


def main() -> None:
    args = parse_args()

    data_yaml = args.data if args.data.is_absolute() else (ROOT / args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"Runs directory not found: {RUNS_DIR}")

    if args.all_images:
        dataset_root = DATASET_YAML.parent
        data_yaml = build_all_images_yaml(dataset_root)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dirs = discover_run_dirs(RUNS_DIR, args.runs)
    if not run_dirs:
        raise RuntimeError(f"No runs with weights/best.pt found in {RUNS_DIR}")

    rows: List[Dict[str, str]] = []

    for run_dir in run_dirs:
        model_path = run_dir / "weights" / "best.pt"
        out_name = f"{run_dir.name}_{args.split}"
        print(f"validating {run_dir.name} -> {model_path.as_posix()}")

        model = YOLO(model_path.as_posix())
        metrics = model.val(
            data=data_yaml.as_posix(),
            imgsz=args.imgsz,
            batch=args.batch,
            split=args.split,
            project=OUT_DIR.as_posix(),
            name=out_name,
            exist_ok=True,
            device=args.device,
            workers=args.workers,
            plots=args.plots,
            verbose=False,
        )

        row = {
            "run": run_dir.name,
            "model": model_path.as_posix(),
            "save_dir": str(getattr(metrics, "save_dir", "")),
            "metrics/mAP50-95(B)": "",
            "metrics/mAP50(B)": "",
            "metrics/precision(B)": "",
            "metrics/recall(B)": "",
            "metrics/mAP50-95(M)": "",
            "metrics/mAP50(M)": "",
            "metrics/precision(M)": "",
            "metrics/recall(M)": "",
        }

        for key in [
            "metrics/mAP50-95(B)",
            "metrics/mAP50(B)",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50-95(M)",
            "metrics/mAP50(M)",
            "metrics/precision(M)",
            "metrics/recall(M)",
        ]:
            value = safe_metric(metrics, key)
            row[key] = "" if value is None else f"{value:.6f}"

        rows.append(row)
        print(
            f"done {run_dir.name}: seg mAP50-95={row['metrics/mAP50-95(M)']} seg mAP50={row['metrics/mAP50(M)']}"
        )

    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "run",
            "model",
            "save_dir",
            "metrics/mAP50-95(B)",
            "metrics/mAP50(B)",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50-95(M)",
            "metrics/mAP50(M)",
            "metrics/precision(M)",
            "metrics/recall(M)",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"summary={SUMMARY_CSV.as_posix()}")


if __name__ == "__main__":
    main()
