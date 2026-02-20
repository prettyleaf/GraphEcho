#!/usr/bin/env python
"""
Resize exported image+polygon dataset and update txt annotations.

Input layout (from tools/export_points_annotations.py):
  <input>/
    <dataset_name>/
      images/*.png
      labels/*.txt
    manifest.csv (optional)

Output:
  <output>/
    <dataset_name>/
      images/*.png
      labels/*.txt
    manifest.csv
    summary.json

TXT annotation format per line:
  class_id x1 y1 x2 y2 ...
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: opencv-python (cv2).") from exc


@dataclass
class TransformParams:
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    out_w: int
    out_h: int


ORIENTATION_MAP_DEFAULT: Dict[str, str] = {
    "cardiac_udc": "rot90_cw",   # left -> up
    "camus": "rot90_ccw",        # right -> up
    "echonet": "rot180",         # down -> up
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize exported images and remap txt polygon points."
    )
    parser.add_argument("--input", type=Path, default=Path("export_points"), help="Input root.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("export_points_800x600"),
        help="Output root.",
    )
    parser.add_argument("--width", type=int, default=800, help="Target image width.")
    parser.add_argument("--height", type=int, default=600, help="Target image height.")
    parser.add_argument(
        "--mode",
        choices=["stretch", "fit"],
        default="stretch",
        help="stretch: force resize to WxH; fit: keep aspect and pad with black.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional dataset folder names to process (default: auto-detect all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max images per dataset for quick test (0 = all).",
    )
    parser.add_argument(
        "--disable-orientation",
        action="store_true",
        help="Disable dataset orientation normalization before resize.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def discover_datasets(input_root: Path) -> List[Path]:
    datasets: List[Path] = []
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "images").is_dir() and (child / "labels").is_dir():
            datasets.append(child)
    return datasets


def load_manifest_index(manifest_path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not manifest_path.exists():
        return {}
    index: Dict[Tuple[str, str], Dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = (row.get("dataset") or "").strip()
            sid = (row.get("sample_id") or "").strip()
            if ds and sid:
                index[(ds, sid)] = row
    return index


def compute_transform(src_w: int, src_h: int, dst_w: int, dst_h: int, mode: str) -> TransformParams:
    if mode == "stretch":
        return TransformParams(
            scale_x=dst_w / float(src_w),
            scale_y=dst_h / float(src_h),
            offset_x=0.0,
            offset_y=0.0,
            out_w=dst_w,
            out_h=dst_h,
        )

    scale = min(dst_w / float(src_w), dst_h / float(src_h))
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))
    offset_x = (dst_w - scaled_w) / 2.0
    offset_y = (dst_h - scaled_h) / 2.0
    return TransformParams(
        scale_x=scale,
        scale_y=scale,
        offset_x=offset_x,
        offset_y=offset_y,
        out_w=dst_w,
        out_h=dst_h,
    )


def resize_image(image, transform: TransformParams, mode: str):
    if mode == "stretch":
        return cv2.resize(image, (transform.out_w, transform.out_h), interpolation=cv2.INTER_LINEAR)

    src_h, src_w = image.shape[:2]
    scaled_w = max(1, int(round(src_w * transform.scale_x)))
    scaled_h = max(1, int(round(src_h * transform.scale_y)))
    resized = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    canvas = 0 * cv2.resize(image, (transform.out_w, transform.out_h), interpolation=cv2.INTER_NEAREST)
    x0 = int(round(transform.offset_x))
    y0 = int(round(transform.offset_y))
    x1 = min(transform.out_w, x0 + scaled_w)
    y1 = min(transform.out_h, y0 + scaled_h)
    canvas[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]
    return canvas


def apply_orientation_to_image(image, orientation: str):
    if orientation == "none":
        return image
    if orientation == "rot90_cw":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == "rot90_ccw":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if orientation == "rot180":
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError(f"Unsupported orientation: {orientation}")


def apply_orientation_to_points(
    points: Sequence[Tuple[float, float]],
    src_w: int,
    src_h: int,
    orientation: str,
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for x, y in points:
        if orientation == "none":
            nx, ny = x, y
        elif orientation == "rot90_cw":
            nx, ny = (src_h - 1) - y, x
        elif orientation == "rot90_ccw":
            nx, ny = y, (src_w - 1) - x
        elif orientation == "rot180":
            nx, ny = (src_w - 1) - x, (src_h - 1) - y
        else:
            raise ValueError(f"Unsupported orientation: {orientation}")
        out.append((nx, ny))
    return out


def parse_annotation_line(line: str) -> Optional[Tuple[str, List[Tuple[float, float]]]]:
    tokens = line.strip().split()
    if len(tokens) < 3:
        return None
    cls = tokens[0]
    coords = tokens[1:]
    if len(coords) % 2 != 0:
        coords = coords[:-1]
    if len(coords) < 6:
        return None

    points: List[Tuple[float, float]] = []
    for i in range(0, len(coords), 2):
        try:
            x = float(coords[i])
            y = float(coords[i + 1])
        except ValueError:
            return None
        points.append((x, y))
    return cls, points


def remap_points(points: Sequence[Tuple[float, float]], t: TransformParams) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for x, y in points:
        nx = x * t.scale_x + t.offset_x
        ny = y * t.scale_y + t.offset_y
        ix = int(round(nx))
        iy = int(round(ny))
        ix = max(0, min(t.out_w - 1, ix))
        iy = max(0, min(t.out_h - 1, iy))
        out.append((ix, iy))
    return out


def unique_point_count(points: Sequence[Tuple[int, int]]) -> int:
    return len(set(points))


def write_annotation(path: Path, polygons: Sequence[Tuple[str, Sequence[Tuple[int, int]]]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for cls, points in polygons:
            coords = " ".join(f"{x} {y}" for x, y in points)
            f.write(f"{cls} {coords}\n")


def process_sample(
    image_in: Path,
    label_in: Path,
    image_out: Path,
    label_out: Path,
    target_w: int,
    target_h: int,
    mode: str,
    orientation: str,
    overwrite: bool,
) -> Tuple[bool, int]:
    if image_out.exists() and label_out.exists() and not overwrite:
        return False, 0

    image = cv2.imread(str(image_in), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_in}")
    src_h, src_w = image.shape[:2]
    image_oriented = apply_orientation_to_image(image, orientation)
    ori_h, ori_w = image_oriented.shape[:2]
    t = compute_transform(ori_w, ori_h, target_w, target_h, mode)
    image_resized = resize_image(image_oriented, t, mode)

    polygons_out: List[Tuple[str, Sequence[Tuple[int, int]]]] = []
    dropped = 0
    if label_in.exists():
        with label_in.open("r", encoding="utf-8") as f:
            for raw in f:
                parsed = parse_annotation_line(raw)
                if parsed is None:
                    continue
                cls, points = parsed
                points_oriented = apply_orientation_to_points(points, src_w, src_h, orientation)
                new_points = remap_points(points_oriented, t)
                if unique_point_count(new_points) < 3:
                    dropped += 1
                    continue
                polygons_out.append((cls, new_points))

    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_out), image_resized):
        raise RuntimeError(f"Failed to write image: {image_out}")
    write_annotation(label_out, polygons_out)
    return True, dropped


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    ds_dirs = discover_datasets(input_root)
    if args.datasets:
        selected = set(args.datasets)
        ds_dirs = [d for d in ds_dirs if d.name in selected]

    manifest_index = load_manifest_index(input_root / "manifest.csv")
    manifest_rows: List[Dict[str, str]] = []

    total_processed = 0
    total_dropped_polygons = 0
    dataset_stats: Dict[str, Dict[str, int]] = {}
    orientation_map = {} if args.disable_orientation else dict(ORIENTATION_MAP_DEFAULT)

    for ds_dir in ds_dirs:
        ds_name = ds_dir.name
        images_in = ds_dir / "images"
        labels_in = ds_dir / "labels"
        images_out = output_root / ds_name / "images"
        labels_out = output_root / ds_name / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        image_files = sorted(images_in.glob("*.png"))
        if args.limit > 0:
            image_files = image_files[: args.limit]

        orientation = orientation_map.get(ds_name, "none")

        processed = 0
        dropped_polygons = 0
        for image_in in image_files:
            sample_id = image_in.stem
            label_in = labels_in / f"{sample_id}.txt"
            image_out = images_out / image_in.name
            label_out = labels_out / f"{sample_id}.txt"

            wrote, dropped = process_sample(
                image_in=image_in,
                label_in=label_in,
                image_out=image_out,
                label_out=label_out,
                target_w=args.width,
                target_h=args.height,
                mode=args.mode,
                orientation=orientation,
                overwrite=args.overwrite,
            )
            dropped_polygons += dropped
            if not wrote:
                continue
            processed += 1
            total_processed += 1

            src_row = manifest_index.get((ds_name, sample_id), {})
            label_lines = 0
            if label_out.exists():
                with label_out.open("r", encoding="utf-8") as f:
                    label_lines = sum(1 for _ in f)

            manifest_rows.append(
                {
                    "dataset": ds_name,
                    "sample_id": sample_id,
                    "source_image": src_row.get("source_image", ""),
                    "source_label": src_row.get("source_label", ""),
                    "image_path": image_out.as_posix(),
                    "label_path": label_out.as_posix(),
                    "num_polygons": str(label_lines),
                    "width": str(args.width),
                    "height": str(args.height),
                    "orientation": orientation,
                }
            )

        total_dropped_polygons += dropped_polygons
        dataset_stats[ds_name] = {
            "processed_images": processed,
            "dropped_polygons": dropped_polygons,
            "orientation": orientation,
        }
        print(
            f"{ds_name}: processed={processed}, dropped_polygons={dropped_polygons}, "
            f"orientation={orientation}"
        )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "dataset",
            "sample_id",
            "source_image",
            "source_label",
            "image_path",
            "label_path",
            "num_polygons",
            "width",
            "height",
            "orientation",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "input_root": input_root.as_posix(),
        "output_root": output_root.as_posix(),
        "target_size": {"width": args.width, "height": args.height},
        "mode": args.mode,
        "orientation_map": orientation_map,
        "total_processed_images": total_processed,
        "total_dropped_polygons": total_dropped_polygons,
        "datasets": dataset_stats,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved manifest: {manifest_path.as_posix()}")
    print(f"Saved summary: {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
