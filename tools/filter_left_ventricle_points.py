#!/usr/bin/env python
"""
Filter exported point annotations to left-ventricle only samples.

Input layout:
  <input>/
    <dataset>/images/*.png
    <dataset>/labels/*.txt
    manifest.csv (optional)

Output layout:
  <output>/
    <dataset>/images/*.png
    <dataset>/labels/*.txt
    manifest.csv
    summary.json

Annotation format:
  class_id x1 y1 x2 y2 ...
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter exported points to left ventricle (class 1).")
    parser.add_argument("--input", type=Path, default=Path("export_points"), help="Input export folder.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("export_points_lv"),
        help="Output folder with LV-only samples.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=1,
        help="Class id to keep (LV is usually class 1).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional dataset folder names to process (default: auto-detect all).",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="How to materialize images in output.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove output folder before writing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output.",
    )
    return parser.parse_args()


def discover_datasets(root: Path) -> List[Path]:
    ds_dirs: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "images").is_dir() and (child / "labels").is_dir():
            ds_dirs.append(child)
    return ds_dirs


def parse_class_id(line: str) -> Optional[int]:
    line = line.strip()
    if not line:
        return None
    token = line.split()[0]
    try:
        return int(float(token))
    except ValueError:
        return None


def load_manifest_index(manifest_path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not manifest_path.exists():
        return {}
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ds = (row.get("dataset") or "").strip()
            sid = (row.get("sample_id") or "").strip()
            if ds and sid:
                out[(ds, sid)] = row
    return out


def copy_image(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if overwrite:
            dst.unlink()
        else:
            return
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ds_dirs = discover_datasets(input_root)
    if args.datasets:
        selected = set(args.datasets)
        ds_dirs = [d for d in ds_dirs if d.name in selected]

    manifest_index = load_manifest_index(input_root / "manifest.csv")
    manifest_rows: List[Dict[str, str]] = []
    summary = {
        "input_root": input_root.as_posix(),
        "output_root": output_root.as_posix(),
        "class_id": args.class_id,
        "datasets": {},
        "total_kept_samples": 0,
    }

    for ds_dir in ds_dirs:
        ds_name = ds_dir.name
        in_images = ds_dir / "images"
        in_labels = ds_dir / "labels"
        out_images = output_root / ds_name / "images"
        out_labels = output_root / ds_name / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        kept = 0
        scanned = 0
        for label_path in sorted(in_labels.glob("*.txt")):
            scanned += 1
            sample_id = label_path.stem
            image_path = in_images / f"{sample_id}.png"
            if not image_path.exists():
                continue

            kept_lines: List[str] = []
            with label_path.open("r", encoding="utf-8") as f:
                for raw in f:
                    cls_id = parse_class_id(raw)
                    if cls_id is None:
                        continue
                    if cls_id == args.class_id:
                        kept_lines.append(raw.strip())

            if not kept_lines:
                continue

            out_img = out_images / image_path.name
            out_lbl = out_labels / label_path.name
            copy_image(image_path, out_img, mode=args.copy_mode, overwrite=args.overwrite)
            write_lines(out_lbl, kept_lines)

            src_row = manifest_index.get((ds_name, sample_id), {})
            manifest_rows.append(
                {
                    "dataset": ds_name,
                    "sample_id": sample_id,
                    "source_image": src_row.get("source_image", ""),
                    "source_label": src_row.get("source_label", ""),
                    "image_path": out_img.as_posix(),
                    "label_path": out_lbl.as_posix(),
                    "num_polygons": str(len(kept_lines)),
                    "class_id": str(args.class_id),
                }
            )
            kept += 1

        summary["datasets"][ds_name] = {
            "scanned_labels": scanned,
            "kept_samples": kept,
        }
        summary["total_kept_samples"] += kept
        print(f"{ds_name}: kept={kept} / scanned={scanned}")

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
            "class_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path.as_posix()}")
    print(f"Saved summary: {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
