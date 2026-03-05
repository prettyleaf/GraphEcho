from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "tools" / "runs"
DATASET_ROOT = ROOT / "export_points_lv_800x600"
OUTPUT_DIR = ROOT / "tools" / "runs" / "inference_preview"
METRIC_COL = "metrics/mAP50-95(M)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image YOLO segmentation inference for local runs.")
    parser.add_argument(
        "--run",
        default="auto",
        help="Run folder name inside tools/runs (e.g. yolo11m-seg_100_m) or 'auto' for best by metric.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to input image. Default: first image from export_points_lv_800x600.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--show", action="store_true", help="Show OpenCV window with prediction.")
    return parser.parse_args()


def resolve_image_path(image_arg: Optional[Path]) -> Path:
    if image_arg is not None:
        p = image_arg if image_arg.is_absolute() else (ROOT / image_arg)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        return p

    for pattern in ("camus/images/*.png", "cardiac_udc/images/*.png", "echonet/images/*.png"):
        found = sorted((DATASET_ROOT / pattern.split("/")[0] / pattern.split("/")[1]).glob("*.png"))
        if found:
            return found[0]

    raise FileNotFoundError("No default image found in export_points_lv_800x600/*/images/*.png")


def load_metric(run_dir: Path) -> float:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return float("-inf")

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return float("-inf")

    values = []
    for row in rows:
        value = row.get(METRIC_COL)
        if value is None or value == "":
            continue
        values.append(float(value))

    return max(values) if values else float("-inf")


def pick_run(run_name: str) -> Path:
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"Runs directory not found: {RUNS_DIR}")

    if run_name != "auto":
        run_dir = RUNS_DIR / run_name
        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"best.pt not found: {best_pt}")
        return run_dir

    candidates = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        best_pt = d / "weights" / "best.pt"
        if not best_pt.exists():
            continue
        candidates.append((load_metric(d), d))

    if not candidates:
        raise FileNotFoundError(f"No run with weights/best.pt found in: {RUNS_DIR}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def preprocess_image(image_path: Path):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if len(img.shape) != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def main() -> None:
    args = parse_args()

    run_dir = pick_run(args.run)
    model_path = run_dir / "weights" / "best.pt"
    image_path = resolve_image_path(args.image)

    model = YOLO(model_path.as_posix())
    img = preprocess_image(image_path)

    results = model(img, conf=args.conf)
    result = results[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"{run_dir.name}_{image_path.stem}_pred.png"
    out_path = OUTPUT_DIR / out_name

    plotted = result.plot()
    cv2.imwrite(str(out_path), plotted)

    print(f"run={run_dir.name}")
    print(f"model={model_path.as_posix()}")
    print(f"image={image_path.as_posix()}")
    print(f"saved={out_path.as_posix()}")

    if args.show:
        cv2.imshow("prediction", plotted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
