#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "tools" / "runs"
VALIDATION_SUMMARY = RUNS_DIR / "validation_runs" / "validation_summary.csv"
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO segmentation on videos and export GIF results.")
    parser.add_argument("--input-dir", type=Path, default=Path("videos"), help="Input folder with videos.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("videos/gif_results"),
        help="Output folder for GIF files.",
    )
    parser.add_argument(
        "--run",
        default="auto",
        help="Run name from tools/runs (e.g. yolo11x-seg_100_x) or 'auto'.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame.")
    parser.add_argument("--gif-fps", type=float, default=12.0, help="GIF frame rate.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap of processed frames per video.")
    return parser.parse_args()


def list_videos(input_dir: Path) -> List[Path]:
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_metric_from_results_csv(results_csv: Path, metric_col: str) -> float:
    if not results_csv.exists():
        return float("-inf")
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return float("-inf")

    values = [safe_float(row.get(metric_col, "")) for row in rows]
    values = [v for v in values if v is not None]
    return max(values) if values else float("-inf")


def candidate_runs(runs_dir: Path) -> Iterable[Path]:
    for d in sorted(runs_dir.iterdir()):
        if d.is_dir() and (d / "weights" / "best.pt").exists():
            yield d


def pick_run_auto(runs_dir: Path) -> Path:
    # Prefer fresh post-training validation summary if present.
    if VALIDATION_SUMMARY.exists():
        with VALIDATION_SUMMARY.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        scored: List[Tuple[float, Path]] = []
        for row in rows:
            run_name = (row.get("run") or "").strip()
            metric = safe_float(row.get("metrics/mAP50-95(M)", ""))
            run_dir = runs_dir / run_name
            if metric is None:
                continue
            if not (run_dir / "weights" / "best.pt").exists():
                continue
            scored.append((metric, run_dir))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

    # Fallback to training metrics from results.csv.
    metric_col = "metrics/mAP50-95(M)"
    scored_fallback: List[Tuple[float, Path]] = []
    for run_dir in candidate_runs(runs_dir):
        score = load_metric_from_results_csv(run_dir / "results.csv", metric_col)
        scored_fallback.append((score, run_dir))
    if not scored_fallback:
        raise FileNotFoundError(f"No runs with weights/best.pt found in {runs_dir}")
    scored_fallback.sort(key=lambda x: x[0], reverse=True)
    return scored_fallback[0][1]


def pick_run(runs_dir: Path, run_name: str) -> Path:
    if run_name != "auto":
        run_dir = runs_dir / run_name
        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            available = sorted(d.name for d in candidate_runs(runs_dir))
            raise FileNotFoundError(
                f"Run not found: {run_name}. Available runs: {', '.join(available)}"
            )
        return run_dir
    return pick_run_auto(runs_dir)


def process_video_to_gif(
    model: YOLO,
    video_path: Path,
    gif_path: Path,
    conf: float,
    frame_step: int,
    gif_fps: float,
    max_frames: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: List[Image.Image] = []
    frame_idx = 0
    kept = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            result = model(frame, conf=conf, verbose=False)[0]
            plotted = result.plot()
            rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            # Palette mode keeps memory usage lower when accumulating frames.
            frames.append(Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=256))

            kept += 1
            frame_idx += 1
            if max_frames > 0 and kept >= max_frames:
                break
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames processed for {video_path.name}")

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / max(gif_fps, 1e-6))))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return kept


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir if args.input_dir.is_absolute() else (ROOT / args.input_dir)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir)
    runs_dir = RUNS_DIR

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be >= 1")

    videos = list_videos(input_dir)
    if not videos:
        raise RuntimeError(f"No video files found in {input_dir}")

    run_dir = pick_run(runs_dir, args.run)
    model_path = run_dir / "weights" / "best.pt"
    model = YOLO(model_path.as_posix())

    print(f"run={run_dir.name}")
    print(f"model={model_path.as_posix()}")
    print(f"videos={len(videos)}")

    for video_path in videos:
        gif_name = f"{video_path.stem}_{run_dir.name}.gif"
        gif_path = output_dir / gif_name
        kept = process_video_to_gif(
            model=model,
            video_path=video_path,
            gif_path=gif_path,
            conf=args.conf,
            frame_step=args.frame_step,
            gif_fps=args.gif_fps,
            max_frames=args.max_frames,
        )
        print(f"done {video_path.name}: frames={kept}, gif={gif_path.as_posix()}")


if __name__ == "__main__":
    main()
