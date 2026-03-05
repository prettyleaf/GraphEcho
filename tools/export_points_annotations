#!/usr/bin/env python
"""
Export image slices/frames and polygon-point annotations (.txt) from:
  - EchoNet-Dynamic
  - CAMUS_public
  - cardiacUDC_dataset

Output layout:
  <output>/
    echonet/{images,labels}
    camus/{images,labels}
    cardiac_udc/{images,labels}
    manifest.csv

Annotation format (.txt, one polygon per line):
  class_id x1 y1 x2 y2 x3 y3 ...
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2
    import nibabel as nib
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency for export script. "
        "Install required packages (opencv-python, nibabel, numpy) and retry."
    ) from exc


Polygon = Tuple[int, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export image + point annotations from all datasets.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("export_points"),
        help="Output folder for exported images/annotations.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["echonet", "camus", "cardiac_udc"],
        choices=["echonet", "camus", "cardiac_udc"],
        help="Datasets to export.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit of exported samples per dataset (0 = no limit).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing image/txt files.",
    )
    return parser.parse_args()


def resolve_dataset_root(root: Path, dataset_name: str) -> Optional[Path]:
    direct = root / dataset_name
    nested = root / "datasets" / dataset_name
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    return None


def ensure_dirs(base: Path) -> Tuple[Path, Path]:
    images = base / "images"
    labels = base / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    return images, labels


def sanitize_id(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def strip_nii_gz(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {arr.shape}")
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - mn) / (mx - mn)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def mask_to_polygons(mask: np.ndarray, min_area: float = 5.0) -> List[Polygon]:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {arr.shape}")
    mask_int = np.rint(arr).astype(np.int32)
    classes = [int(c) for c in np.unique(mask_int) if int(c) > 0]

    polygons: List[Polygon] = []
    for cls_id in classes:
        binary = (mask_int == cls_id).astype(np.uint8)
        if binary.max() == 0:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            points = contour[:, 0, :].astype(np.int32)
            if np.unique(points, axis=0).shape[0] < 3:
                continue
            polygons.append((cls_id, points))
    return polygons


def write_polygons_txt(path: Path, polygons: Sequence[Polygon]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for cls_id, points in polygons:
            coords = " ".join(f"{int(x)} {int(y)}" for x, y in points.tolist())
            f.write(f"{cls_id} {coords}\n")


def safe_load_nifti(path: Path) -> Optional[np.ndarray]:
    try:
        return np.asarray(nib.load(str(path)).dataobj)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] skip broken nifti: {path} ({exc})")
        return None


def save_sample(
    image: np.ndarray,
    polygons: Sequence[Polygon],
    image_path: Path,
    label_path: Path,
    overwrite: bool,
) -> bool:
    if image_path.exists() and label_path.exists() and not overwrite:
        return False

    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    img_u8 = to_uint8(image)
    ok = cv2.imwrite(str(image_path), img_u8)
    if not ok:
        raise RuntimeError(f"Failed to write image: {image_path}")
    write_polygons_txt(label_path, polygons)
    return True


def tracing_rows_to_polygon(rows: Sequence[Tuple[float, float, float, float]]) -> Optional[np.ndarray]:
    if len(rows) < 2:
        return None
    left = np.array([[r[0], r[1]] for r in rows], dtype=np.float32)
    right = np.array([[r[2], r[3]] for r in rows], dtype=np.float32)
    polygon = np.concatenate([left, right[::-1]], axis=0)
    polygon = np.rint(polygon).astype(np.int32)
    if np.unique(polygon, axis=0).shape[0] < 3:
        return None
    return polygon


def export_echonet(
    root: Path,
    out_root: Path,
    limit: int,
    overwrite: bool,
    manifest_rows: List[Dict[str, str]],
) -> int:
    dataset_root = resolve_dataset_root(root, "EchoNet-Dynamic")
    if dataset_root is None:
        print("Skip EchoNet: dataset not found.")
        return 0

    videos_root = dataset_root / "Videos"
    tracings_csv = dataset_root / "VolumeTracings.csv"
    if not tracings_csv.exists():
        print(f"Skip EchoNet: missing {tracings_csv}")
        return 0

    out_images, out_labels = ensure_dirs(out_root / "echonet")

    grouped: Dict[str, Dict[int, List[Tuple[float, float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with tracings_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("FileName") or "").strip()
            if not name:
                continue
            if not name.lower().endswith(".avi"):
                name += ".avi"
            try:
                frame = int(float(row["Frame"]))
                x1 = float(row["X1"])
                y1 = float(row["Y1"])
                x2 = float(row["X2"])
                y2 = float(row["Y2"])
            except (TypeError, ValueError, KeyError):
                continue
            grouped[name][frame].append((x1, y1, x2, y2))

    exported = 0
    for video_name in sorted(grouped.keys()):
        video_path = videos_root / video_name
        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        frame_dict = grouped[video_name]
        for frame_idx in sorted(frame_dict.keys()):
            polygon = tracing_rows_to_polygon(frame_dict[frame_idx])
            if polygon is None:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok and frame_idx > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
                ok, frame = cap.read()
            if not ok:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            polygons: List[Polygon] = [(1, polygon)]
            sample_id = sanitize_id(f"{Path(video_name).stem}_f{frame_idx:04d}")
            image_path = out_images / f"{sample_id}.png"
            label_path = out_labels / f"{sample_id}.txt"

            wrote = save_sample(gray, polygons, image_path, label_path, overwrite=overwrite)
            if wrote:
                exported += 1
                manifest_rows.append(
                    {
                        "dataset": "echonet",
                        "sample_id": sample_id,
                        "source_image": video_path.as_posix(),
                        "source_label": tracings_csv.as_posix(),
                        "image_path": image_path.as_posix(),
                        "label_path": label_path.as_posix(),
                        "num_polygons": str(len(polygons)),
                    }
                )
                if limit > 0 and exported >= limit:
                    cap.release()
                    return exported

        cap.release()
    return exported


def iter_slices(arr: np.ndarray) -> Iterable[Tuple[int, np.ndarray]]:
    if arr.ndim == 2:
        yield 0, arr
    elif arr.ndim == 3:
        for i in range(arr.shape[-1]):
            yield i, arr[..., i]
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")


def export_camus(
    root: Path,
    out_root: Path,
    limit: int,
    overwrite: bool,
    manifest_rows: List[Dict[str, str]],
) -> int:
    dataset_root = resolve_dataset_root(root, "CAMUS_public")
    if dataset_root is None:
        print("Skip CAMUS: dataset not found.")
        return 0

    nifti_root = dataset_root / "database_nifti"
    if not nifti_root.exists():
        print(f"Skip CAMUS: missing {nifti_root}")
        return 0

    out_images, out_labels = ensure_dirs(out_root / "camus")
    mask_paths = sorted(nifti_root.rglob("*_gt.nii.gz"))

    exported = 0
    for mask_path in mask_paths:
        image_path_src = Path(str(mask_path).replace("_gt.nii.gz", ".nii.gz"))
        if not image_path_src.exists():
            continue

        image_arr = safe_load_nifti(image_path_src)
        mask_arr = safe_load_nifti(mask_path)
        if image_arr is None or mask_arr is None:
            continue
        if image_arr.shape != mask_arr.shape:
            continue

        rel_id = sanitize_id(str(image_path_src.relative_to(dataset_root)).replace("\\", "/"))
        rel_id = rel_id.replace(".nii.gz", "")

        for slice_idx, image_slice in iter_slices(image_arr):
            mask_slice = mask_arr if mask_arr.ndim == 2 else mask_arr[..., slice_idx]
            polygons = mask_to_polygons(mask_slice)
            sample_id = f"{rel_id}_s{slice_idx:03d}" if image_arr.ndim == 3 else rel_id

            image_path = out_images / f"{sample_id}.png"
            label_path = out_labels / f"{sample_id}.txt"
            wrote = save_sample(image_slice, polygons, image_path, label_path, overwrite=overwrite)
            if wrote:
                exported += 1
                manifest_rows.append(
                    {
                        "dataset": "camus",
                        "sample_id": sample_id,
                        "source_image": image_path_src.as_posix(),
                        "source_label": mask_path.as_posix(),
                        "image_path": image_path.as_posix(),
                        "label_path": label_path.as_posix(),
                        "num_polygons": str(len(polygons)),
                    }
                )
                if limit > 0 and exported >= limit:
                    return exported
    return exported


def export_cardiac_udc(
    root: Path,
    out_root: Path,
    limit: int,
    overwrite: bool,
    manifest_rows: List[Dict[str, str]],
) -> int:
    dataset_root = resolve_dataset_root(root, "cardiacUDC_dataset")
    if dataset_root is None:
        print("Skip CardiacUDA: dataset not found.")
        return 0

    out_images, out_labels = ensure_dirs(out_root / "cardiac_udc")
    image_paths = sorted(dataset_root.rglob("*_image.nii.gz"))

    exported = 0
    for image_path_src in image_paths:
        label_path_src = image_path_src.with_name(image_path_src.name.replace("_image.nii.gz", "_label.nii.gz"))
        if not label_path_src.exists():
            continue

        image_arr = safe_load_nifti(image_path_src)
        label_arr = safe_load_nifti(label_path_src)
        if image_arr is None or label_arr is None:
            continue
        if image_arr.shape != label_arr.shape:
            continue

        rel = str(image_path_src.relative_to(dataset_root)).replace("\\", "/")
        rel_id = sanitize_id(rel.replace(".nii.gz", ""))

        for slice_idx, image_slice in iter_slices(image_arr):
            label_slice = label_arr if label_arr.ndim == 2 else label_arr[..., slice_idx]
            polygons = mask_to_polygons(label_slice)
            sample_id = f"{rel_id}_s{slice_idx:03d}" if image_arr.ndim == 3 else rel_id

            image_path = out_images / f"{sample_id}.png"
            label_path = out_labels / f"{sample_id}.txt"
            wrote = save_sample(image_slice, polygons, image_path, label_path, overwrite=overwrite)
            if wrote:
                exported += 1
                manifest_rows.append(
                    {
                        "dataset": "cardiac_udc",
                        "sample_id": sample_id,
                        "source_image": image_path_src.as_posix(),
                        "source_label": label_path_src.as_posix(),
                        "image_path": image_path.as_posix(),
                        "label_path": label_path.as_posix(),
                        "num_polygons": str(len(polygons)),
                    }
                )
                if limit > 0 and exported >= limit:
                    return exported
    return exported


def write_manifest(manifest_rows: Sequence[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "sample_id",
        "source_image",
        "source_label",
        "image_path",
        "label_path",
        "num_polygons",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_root = args.output.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []
    results: Dict[str, int] = {}

    if "echonet" in args.datasets:
        results["echonet"] = export_echonet(root, out_root, args.limit, args.overwrite, manifest_rows)
    if "camus" in args.datasets:
        results["camus"] = export_camus(root, out_root, args.limit, args.overwrite, manifest_rows)
    if "cardiac_udc" in args.datasets:
        results["cardiac_udc"] = export_cardiac_udc(
            root, out_root, args.limit, args.overwrite, manifest_rows
        )

    write_manifest(manifest_rows, out_root / "manifest.csv")

    print("Export finished:")
    for name, count in results.items():
        print(f"  {name}: {count}")
    print(f"  manifest: {(out_root / 'manifest.csv').as_posix()}")


if __name__ == "__main__":
    main()
