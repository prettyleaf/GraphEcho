#!/usr/bin/env python
"""
Train Ultralytics YOLO segmentation on exported point labels.

Features:
- YOLO11-seg default model (yolo11m-seg.pt)
- Mosaic disabled
- 100 epochs by default
- Time-based early stop: stop if no improvement for N minutes
- GPU usage logging (+ optional soft throttle if utilization > threshold)
- New unique run folder per launch (configurable)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:
    import cv2  # type: ignore
except ModuleNotFoundError:
    cv2 = None

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Sample:
    dataset: str
    sample_id: str
    image: Path
    label: Optional[Path]


BatchValue = Union[int, float]


def parse_batch_value(raw_value: str) -> BatchValue:
    raw = raw_value.strip().lower()
    if raw == "auto":
        return -1
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "batch must be int, 'auto', or fraction in (0, 1), e.g. 8 / auto / 0.95"
        ) from exc

    if value == -1.0:
        return -1
    if 0.0 < value < 1.0:
        return float(value)
    if value >= 1.0 and value.is_integer():
        return int(value)
    raise argparse.ArgumentTypeError(
        "batch must be int>=1, -1/auto, or fraction in (0, 1)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11/YOLO12 segmentation with controlled GPU load.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("export_points_lv_800x600"),
        help="Input folder with per-dataset images/labels.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs_yolo_seg"),
        help="Root folder for training runs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="yolo_seg",
        help="Base run name.",
    )
    parser.add_argument(
        "--new-run-folder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a unique folder each run (default: true).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional dataset folder names to include (default: auto-detect all).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m-seg.pt",
        help="Ultralytics model. Example: yolo11m-seg.pt",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs.")
    parser.add_argument("--imgsz", type=int, default=800, help="Image size for YOLO.")
    parser.add_argument(
        "--batch",
        type=parse_batch_value,
        default=0.95,
        help="Batch: int (e.g. 8), -1/auto (autobatch 60%%), or fraction (e.g. 0.95 for 95%% VRAM target).",
    )
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default="0", help="Device string (e.g. 0, 0,1, cpu).")
    parser.add_argument(
        "--allow-cpu-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If CUDA device is requested but unavailable, fall back to CPU instead of failing.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--materialize",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="How to materialize image files into YOLO dataset folder.",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="lv",
        help="Class name if single class after remap.",
    )
    parser.add_argument(
        "--no-improve-minutes",
        type=float,
        default=15.0,
        help="Stop training if no metric improvement for this many minutes.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-6,
        help="Minimum fitness delta to count as improvement.",
    )
    parser.add_argument(
        "--max-gpu-util",
        type=float,
        default=95.0,
        help="Soft GPU util threshold in percent. <=0 disables throttle.",
    )
    parser.add_argument(
        "--max-vram-util",
        type=float,
        default=95.0,
        help="Soft VRAM usage threshold in percent of total memory. <=0 disables.",
    )
    parser.add_argument(
        "--max-mem-ctrl-util",
        type=float,
        default=95.0,
        help="Soft memory-controller utilization threshold (nvidia-smi utilization.memory). <=0 disables.",
    )
    parser.add_argument(
        "--max-power-util",
        type=float,
        default=95.0,
        help="Soft board power threshold in percent of power limit. <=0 disables.",
    )
    parser.add_argument(
        "--gpu-log-interval",
        type=float,
        default=10.0,
        help="Seconds between GPU usage logs.",
    )
    parser.add_argument(
        "--throttle-sleep",
        type=float,
        default=0.5,
        help="Sleep duration when throttling for high GPU utilization.",
    )
    parser.add_argument(
        "--max-throttle-loops",
        type=int,
        default=20,
        help="Maximum throttle loops per batch when limits are exceeded.",
    )
    parser.add_argument(
        "--strict-gpu-limits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop training when GPU/VRAM/power limits keep being exceeded after throttle.",
    )
    parser.add_argument(
        "--limit-grace-count",
        type=int,
        default=3,
        help="Consecutive post-throttle breaches allowed before strict stop.",
    )
    parser.add_argument(
        "--auto-reduce-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After strict stop by limits, reduce batch and retry automatically.",
    )
    parser.add_argument(
        "--batch-reduce-factor",
        type=float,
        default=0.85,
        help="Batch reduction factor for automatic retries after strict limit stop.",
    )
    parser.add_argument(
        "--min-batch-fraction",
        type=float,
        default=0.50,
        help="Minimum allowed batch fraction when batch is fractional.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum train attempts when auto-reducing batch.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="ram",
        help="Ultralytics cache mode: False/True/ram/disk.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable AMP mixed precision.",
    )
    return parser.parse_args()


def discover_dataset_dirs(input_root: Path) -> List[Path]:
    dirs: List[Path] = []
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "images").is_dir() and (child / "labels").is_dir():
            dirs.append(child)
    return dirs


def iter_image_files(images_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            files.append(p)
    return sorted(files)


def collect_samples(input_root: Path, include_datasets: Sequence[str]) -> List[Sample]:
    ds_dirs = discover_dataset_dirs(input_root)
    if include_datasets:
        allowed = set(include_datasets)
        ds_dirs = [d for d in ds_dirs if d.name in allowed]
    if not ds_dirs:
        raise RuntimeError("No dataset folders with images/labels found in input.")

    samples: List[Sample] = []
    for ds_dir in ds_dirs:
        ds_name = ds_dir.name
        images_dir = ds_dir / "images"
        labels_dir = ds_dir / "labels"
        for image_path in iter_image_files(images_dir):
            sid = image_path.stem
            label_path = labels_dir / f"{sid}.txt"
            samples.append(
                Sample(
                    dataset=ds_name,
                    sample_id=sid,
                    image=image_path,
                    label=label_path if label_path.exists() else None,
                )
            )
    if not samples:
        raise RuntimeError("No image samples found.")
    return samples


def split_samples(samples: Sequence[Sample], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[Sample]]:
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be > 0")
    if train_ratio + val_ratio > 1.0 + 1e-9:
        raise ValueError("train_ratio + val_ratio must be <= 1")

    grouped: Dict[str, List[Sample]] = {}
    for s in samples:
        grouped.setdefault(s.dataset, []).append(s)

    rnd = random.Random(seed)
    train: List[Sample] = []
    val: List[Sample] = []
    test: List[Sample] = []

    for ds_name, ds_samples in grouped.items():
        items = ds_samples[:]
        rnd.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n_train == 0 and n > 0:
            n_train = 1
        if n_val == 0 and (n - n_train) > 1:
            n_val = 1

        ds_train = items[:n_train]
        ds_val = items[n_train : n_train + n_val]
        ds_test = items[n_train + n_val :]

        train.extend(ds_train)
        val.extend(ds_val)
        test.extend(ds_test)

    if not val:
        raise RuntimeError("Validation split is empty. Increase val_ratio or dataset size.")
    return {"train": train, "val": val, "test": test}


def parse_label_line(line: str) -> Optional[Tuple[int, List[float]]]:
    tokens = line.strip().split()
    if len(tokens) < 7:
        return None
    try:
        cls_raw = int(float(tokens[0]))
    except ValueError:
        return None

    coords: List[float] = []
    for t in tokens[1:]:
        try:
            coords.append(float(t))
        except ValueError:
            return None
    if len(coords) % 2 != 0:
        coords = coords[:-1]
    if len(coords) < 6:
        return None
    return cls_raw, coords


def collect_class_ids(samples: Sequence[Sample]) -> List[int]:
    class_ids = set()
    for s in samples:
        if s.label is None or not s.label.exists():
            continue
        with s.label.open("r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_label_line(line)
                if parsed is None:
                    continue
                class_ids.add(parsed[0])
    if not class_ids:
        class_ids.add(0)
    return sorted(class_ids)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def convert_label_to_yolo(
    src_label: Optional[Path],
    dst_label: Path,
    img_w: int,
    img_h: int,
    class_map: Dict[int, int],
) -> int:
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    lines_out: List[str] = []

    if src_label is not None and src_label.exists():
        with src_label.open("r", encoding="utf-8") as f:
            for raw in f:
                parsed = parse_label_line(raw)
                if parsed is None:
                    continue
                cls_raw, coords = parsed
                if cls_raw not in class_map:
                    continue
                cls_new = class_map[cls_raw]

                is_normalized = all(0.0 <= c <= 1.5 for c in coords)
                out_coords: List[float] = []
                for i in range(0, len(coords), 2):
                    x = coords[i]
                    y = coords[i + 1]
                    if is_normalized:
                        nx = min(1.0, max(0.0, x))
                        ny = min(1.0, max(0.0, y))
                    else:
                        px = min(float(img_w - 1), max(0.0, x))
                        py = min(float(img_h - 1), max(0.0, y))
                        nx = px / float(max(1, img_w))
                        ny = py / float(max(1, img_h))
                    out_coords.extend([nx, ny])

                if len(out_coords) < 6:
                    continue
                coord_text = " ".join(f"{v:.6f}" for v in out_coords)
                lines_out.append(f"{cls_new} {coord_text}")
                written += 1

    with dst_label.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines_out:
            f.write(line + "\n")
    return written


def read_image_size(image_path: Path) -> Tuple[int, int]:
    if cv2 is not None:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            h, w = img.shape[:2]
            return w, h

    if Image is not None:
        with Image.open(image_path) as im:
            w, h = im.size
            return w, h

    raise RuntimeError(
        "Cannot read image size. Install one of: opencv-python, pillow"
    )


def build_yolo_dataset(
    run_dir: Path,
    split: Dict[str, List[Sample]],
    class_map: Dict[int, int],
    class_names: List[str],
    materialize: str,
) -> Path:
    yolo_root = run_dir / "dataset_yolo"
    for part in ("train", "val", "test"):
        (yolo_root / "images" / part).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / part).mkdir(parents=True, exist_ok=True)

    for part, items in split.items():
        for idx, s in enumerate(items):
            try:
                w, h = read_image_size(s.image)
            except Exception as exc:
                print(f"[warn] failed to read image '{s.image.as_posix()}': {exc}")
                continue
            ext = s.image.suffix.lower()
            new_id = f"{s.dataset}_{s.sample_id}_{idx:07d}"
            img_dst = yolo_root / "images" / part / f"{new_id}{ext}"
            lbl_dst = yolo_root / "labels" / part / f"{new_id}.txt"

            link_or_copy(s.image, img_dst, mode=materialize)
            convert_label_to_yolo(s.label, lbl_dst, img_w=w, img_h=h, class_map=class_map)

    yaml_path = run_dir / "dataset.yaml"
    lines = [
        f"path: {yolo_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, n in enumerate(class_names):
        lines.append(f"  {i}: {n}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def parse_gpu_index(device: str) -> Optional[int]:
    d = device.strip().lower()
    if d in {"cpu", "mps"}:
        return None
    first = d.split(",")[0].strip()
    if first.isdigit():
        return int(first)
    return 0


def parse_smi_float(raw: str) -> Optional[float]:
    token = raw.strip()
    if not token:
        return None
    upper = token.upper()
    if upper in {"N/A", "[N/A]", "NA"}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def resolve_training_device(requested_device: str, allow_cpu_fallback: bool) -> str:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed in this environment. Install torch before training."
        ) from exc

    d = requested_device.strip().lower()
    if d in {"cpu", "mps"}:
        return requested_device

    cuda_ok = bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0
    if cuda_ok:
        return requested_device

    msg = (
        "CUDA device requested but torch CUDA is unavailable. "
        f"torch.cuda.is_available()={torch.cuda.is_available()}, "
        f"torch.cuda.device_count()={torch.cuda.device_count()}."
    )
    if allow_cpu_fallback:
        print(f"[warn] {msg} Falling back to device=cpu.")
        return "cpu"
    raise SystemExit(msg)


def query_gpu(gpu_index: int) -> Optional[Dict[str, float]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu_index),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=2.0).strip()
    except Exception:
        return None
    if not out:
        return None
    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 7:
        return None
    util = parse_smi_float(parts[0])
    mem_util = parse_smi_float(parts[1])
    mem_used = parse_smi_float(parts[2])
    mem_total = parse_smi_float(parts[3])
    temp = parse_smi_float(parts[4])
    power_draw = parse_smi_float(parts[5])
    power_limit = parse_smi_float(parts[6])
    if util is None or mem_used is None or mem_total is None:
        return None
    mem_pct = (100.0 * mem_used / mem_total) if mem_total > 0 else 0.0
    power_pct = None
    if power_draw is not None and power_limit is not None and power_limit > 0:
        power_pct = 100.0 * power_draw / power_limit
    return {
        "util": util,
        "mem_util": mem_util if mem_util is not None else -1.0,
        "mem_used": mem_used,
        "mem_total": mem_total,
        "mem_pct": mem_pct,
        "temp": temp if temp is not None else -1.0,
        "power_draw": power_draw if power_draw is not None else -1.0,
        "power_limit": power_limit if power_limit is not None else -1.0,
        "power_pct": power_pct if power_pct is not None else -1.0,
    }


def format_batch_value(batch: BatchValue) -> str:
    if isinstance(batch, int):
        if batch == -1:
            return "auto(-1)"
        return str(batch)
    return f"{batch:.3f}"


def reduce_batch_value(batch: BatchValue, factor: float, min_fraction: float) -> Optional[BatchValue]:
    safe_factor = min(0.95, max(0.10, float(factor)))
    if isinstance(batch, int):
        if batch == -1:
            return 0.90
        if batch <= 1:
            return None
        new_batch = int(max(1, round(batch * safe_factor)))
        if new_batch >= batch:
            new_batch = batch - 1
        return new_batch if new_batch >= 1 else None

    frac = float(batch)
    if not (0.0 < frac < 1.0):
        return None
    floor = max(0.05, min(0.99, float(min_fraction)))
    new_frac = max(floor, frac * safe_factor)
    if new_frac >= frac:
        new_frac = max(floor, frac - 0.05)
    if new_frac <= 0.0 or new_frac >= frac:
        return None
    return round(new_frac, 3)


def extract_fitness(trainer) -> Optional[float]:
    fit = getattr(trainer, "fitness", None)
    if fit is not None:
        try:
            return float(fit)
        except Exception:
            pass

    metrics = getattr(trainer, "metrics", None)
    if isinstance(metrics, dict):
        preferred = [
            "metrics/seg/mAP50-95(B)",
            "metrics/seg/mAP50-95(M)",
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95(M)",
            "metrics/mAP50-95",
        ]
        for k in preferred:
            if k in metrics:
                try:
                    return float(metrics[k])
                except Exception:
                    pass
        vals = [v for v in metrics.values() if isinstance(v, (int, float))]
        if vals:
            return float(max(vals))
    return None


def make_run_dir(output_root: Path, run_name: str, new_run_folder: bool) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if not new_run_folder:
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_root / f"{run_name}_{ts}"
    run_dir = base
    idx = 1
    while run_dir.exists():
        run_dir = output_root / f"{base.name}_{idx:02d}"
        idx += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")
    if cv2 is None and Image is None:
        raise SystemExit("Need image reader dependency: pip install opencv-python or pillow")

    run_dir = make_run_dir(args.output_root.resolve(), args.run_name, args.new_run_folder)
    print(f"Run dir: {run_dir.as_posix()}")

    samples = collect_samples(input_root, args.datasets)
    split = split_samples(samples, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    class_ids = collect_class_ids(samples)
    class_map = {orig: i for i, orig in enumerate(class_ids)}
    if len(class_ids) == 1:
        class_names = [args.class_name]
    else:
        class_names = [f"class_{orig}" for orig in class_ids]

    dataset_yaml = build_yolo_dataset(
        run_dir=run_dir,
        split=split,
        class_map=class_map,
        class_names=class_names,
        materialize=args.materialize,
    )
    print(f"Prepared dataset yaml: {dataset_yaml.as_posix()}")
    print(
        f"Split sizes: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}; "
        f"classes={class_map}"
    )

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install with: pip install ultralytics"
        ) from exc

    train_device = resolve_training_device(args.device, args.allow_cpu_fallback)
    gpu_index = parse_gpu_index(train_device)
    train_batch: BatchValue = args.batch
    max_attempts = max(1, int(args.max_retries))
    grace_count = max(1, int(args.limit_grace_count))
    max_throttle_loops = max(1, int(args.max_throttle_loops))

    config_dump = {
        "args": vars(args),
        "run_dir": run_dir.as_posix(),
        "dataset_yaml": dataset_yaml.as_posix(),
        "class_map": class_map,
        "class_names": class_names,
        "split_sizes": {k: len(v) for k, v in split.items()},
        "resolved_device": train_device,
        "initial_batch": train_batch,
    }
    (run_dir / "train_config.json").write_text(
        json.dumps(config_dump, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    no_improve_seconds = float(args.no_improve_minutes) * 60.0

    def _limit_reasons(stat: Dict[str, float]) -> List[str]:
        reasons: List[str] = []
        if args.max_gpu_util > 0 and stat["util"] > args.max_gpu_util:
            reasons.append(f"gpu_util={stat['util']:.1f}%>{args.max_gpu_util:.1f}%")
        if args.max_vram_util > 0 and stat["mem_pct"] > args.max_vram_util:
            reasons.append(f"vram={stat['mem_pct']:.1f}%>{args.max_vram_util:.1f}%")
        mem_ctrl = stat.get("mem_util", -1.0)
        if args.max_mem_ctrl_util > 0 and mem_ctrl >= 0 and mem_ctrl > args.max_mem_ctrl_util:
            reasons.append(f"mem_ctrl={mem_ctrl:.1f}%>{args.max_mem_ctrl_util:.1f}%")
        power_pct = stat.get("power_pct", -1.0)
        if args.max_power_util > 0 and power_pct >= 0 and power_pct > args.max_power_util:
            reasons.append(f"power={power_pct:.1f}%>{args.max_power_util:.1f}%")
        return reasons

    def _gpu_log(prefix: str = "gpu", stat: Optional[Dict[str, float]] = None) -> None:
        if gpu_index is None:
            return
        live = stat if stat is not None else query_gpu(gpu_index)
        if live is None:
            print(f"[{prefix}] nvidia-smi unavailable")
            return
        msg = (
            f"[{prefix}] util={live['util']:.1f}% "
            f"vram={live['mem_used']:.0f}/{live['mem_total']:.0f}MB ({live['mem_pct']:.1f}%)"
        )
        if live.get("mem_util", -1.0) >= 0:
            msg += f" mem_ctrl={live['mem_util']:.1f}%"
        if live.get("power_pct", -1.0) >= 0 and live.get("power_draw", -1.0) >= 0 and live.get("power_limit", -1.0) > 0:
            msg += f" power={live['power_draw']:.1f}/{live['power_limit']:.1f}W ({live['power_pct']:.1f}%)"
        if live.get("temp", -1.0) >= 0:
            msg += f" temp={live['temp']:.1f}C"
        print(msg)

    # Use project/name from run_dir to keep each run isolated.
    project = run_dir.parent
    base_name = run_dir.name
    final_run_name = base_name

    attempt = 1
    while True:
        attempt_name = base_name if attempt == 1 else f"{base_name}_retry{attempt - 1:02d}"
        final_run_name = attempt_name
        print(
            f"Train attempt {attempt}/{max_attempts} | device={train_device} | "
            f"batch={format_batch_value(train_batch)}"
        )

        model = YOLO(args.model)
        cb_state = {
            "best": float("-inf"),
            "last_improve": time.time(),
            "last_gpu_log": 0.0,
            "limit_breach_count": 0,
            "stopped_by_limits": False,
            "stop_reason": "",
        }

        def on_train_start(trainer) -> None:
            cb_state["last_improve"] = time.time()
            cb_state["best"] = float("-inf")
            cb_state["last_gpu_log"] = 0.0
            cb_state["limit_breach_count"] = 0
            cb_state["stopped_by_limits"] = False
            cb_state["stop_reason"] = ""
            print(
                f"Time-stop if no improvement for {args.no_improve_minutes:.1f} min. "
                f"Limits: gpu<={args.max_gpu_util:.1f}%, vram<={args.max_vram_util:.1f}%, "
                f"mem_ctrl<={args.max_mem_ctrl_util:.1f}%, power<={args.max_power_util:.1f}%"
            )
            print(f"Training device: {train_device}")
            _gpu_log("train_start")

        def on_train_batch_end(trainer) -> None:
            now = time.time()
            if gpu_index is None:
                return

            stat = query_gpu(gpu_index)
            if stat is None:
                return

            if (now - cb_state["last_gpu_log"]) >= float(args.gpu_log_interval):
                _gpu_log("batch", stat)
                cb_state["last_gpu_log"] = now

            reasons = _limit_reasons(stat)
            if not reasons:
                cb_state["limit_breach_count"] = 0
                return

            loops = 0
            while reasons and loops < max_throttle_loops:
                print(f"[throttle] {'; '.join(reasons)} -> sleep {args.throttle_sleep:.2f}s")
                time.sleep(float(args.throttle_sleep))
                loops += 1
                stat = query_gpu(gpu_index)
                if stat is None:
                    reasons = []
                    break
                reasons = _limit_reasons(stat)

            if not reasons:
                cb_state["limit_breach_count"] = 0
                return

            cb_state["limit_breach_count"] += 1
            print(
                f"[limit] still above after throttle ({cb_state['limit_breach_count']}/{grace_count}): "
                f"{'; '.join(reasons)}"
            )
            if bool(args.strict_gpu_limits) and cb_state["limit_breach_count"] >= grace_count:
                cb_state["stopped_by_limits"] = True
                cb_state["stop_reason"] = "; ".join(reasons)
                print(f"[limit-stop] {cb_state['stop_reason']}")
                setattr(trainer, "stop", True)

        def on_fit_epoch_end(trainer) -> None:
            now = time.time()
            score = extract_fitness(trainer)
            if score is not None and (score > cb_state["best"] + float(args.min_delta)):
                cb_state["best"] = score
                cb_state["last_improve"] = now
                print(f"[improve] fitness={score:.6f}")

            silence = now - cb_state["last_improve"]
            if silence >= no_improve_seconds:
                mins = silence / 60.0
                print(
                    f"[early-stop] no improvement for {mins:.1f} min "
                    f"(limit {args.no_improve_minutes:.1f}). stopping."
                )
                setattr(trainer, "stop", True)

        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_batch_end", on_train_batch_end)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        model.train(
            data=dataset_yaml.as_posix(),
            task="segment",
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=train_batch,
            workers=args.workers,
            device=train_device,
            project=project.as_posix(),
            name=attempt_name,
            exist_ok=True,
            mosaic=0.0,
            close_mosaic=0,
            patience=args.epochs + 1,  # disable epoch-based early stop; use time-based callback.
            amp=bool(args.amp),
            cache=args.cache,
            pretrained=True,
            verbose=True,
            seed=args.seed,
        )

        if not cb_state["stopped_by_limits"]:
            break

        if (not bool(args.auto_reduce_batch)) or attempt >= max_attempts:
            print("[stop] strict limits reached and retry is disabled or max retries reached.")
            break

        next_batch = reduce_batch_value(train_batch, factor=args.batch_reduce_factor, min_fraction=args.min_batch_fraction)
        if next_batch is None:
            print("[stop] strict limits reached and batch can not be reduced further.")
            break
        print(
            f"[retry] strict limits were exceeded. "
            f"Retrying with reduced batch {format_batch_value(train_batch)} -> {format_batch_value(next_batch)}"
        )
        train_batch = next_batch
        attempt += 1

    print("Training finished.")
    print(f"Run outputs: {(project / final_run_name).as_posix()}")


if __name__ == "__main__":
    main()
