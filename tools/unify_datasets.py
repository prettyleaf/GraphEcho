#!/usr/bin/env python
"""
Unify heterogeneous echo datasets into one manifest schema.

Supported dataset roots under --root:
  - EchoNet-Dynamic (or datasets/EchoNet-Dynamic)
  - CAMUS_public (or datasets/CAMUS_public)
  - cardiacUDC_dataset (or datasets/cardiacUDC_dataset)

Output:
  - <output>/manifest.csv
  - <output>/summary.json
  - optional materialized files under <output>/data/ (copy/link/symlink)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CAMUS_FILE_RE = re.compile(
    r"^(?P<patient>patient\d+)_(?P<view>2CH|4CH)_(?P<phase>ED|ES|half_sequence)(?P<is_gt>_gt)?\.nii\.gz$"
)
CARDIAC_UDC_IMAGE_RE = re.compile(
    r"^(?P<subject>[A-Za-z]+)-(?P<subject_id>\d+)-(?P<view>\d+)_image\.nii\.gz$"
)
ECHO_SPLIT_MAP = {"TRAIN": "train", "VAL": "val", "TEST": "test"}


@dataclass
class UnifiedRecord:
    sample_id: str
    dataset: str
    split: str
    task: str
    modality: str
    source_image: str
    source_label: str
    unified_image: str
    unified_label: str
    patient_id: str
    view: str
    phase: str
    site: str
    target_ef: str
    target_esv: str
    target_edv: str
    fps: str
    num_frames: str
    frame_height: str
    frame_width: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified dataset manifest from multiple source datasets."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root containing dataset folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("unified_dataset"),
        help="Output directory for unified artifacts.",
    )
    parser.add_argument(
        "--mode",
        choices=["manifest", "hardlink", "symlink", "copy"],
        default="manifest",
        help="How to materialize files into output/data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output/data.",
    )
    return parser.parse_args()


def rel_or_abs(path: Path, root: Path) -> str:
    if not path:
        return ""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def normalize_split(raw: str) -> str:
    value = (raw or "").strip()
    return ECHO_SPLIT_MAP.get(value.upper(), value.lower() or "unknown")


def suffix_for(path: Path) -> str:
    lowered = path.name.lower()
    if lowered.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix or ""


def safe_sample_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def resolve_dataset_root(root: Path, dataset_name: str) -> Path:
    direct = root / dataset_name
    nested = root / "datasets" / dataset_name
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    return nested


def transfer_file(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if overwrite:
            dst.unlink()
        else:
            return

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def build_echonet_records(root: Path) -> List[UnifiedRecord]:
    dataset_root = resolve_dataset_root(root, "EchoNet-Dynamic")
    file_list_path = dataset_root / "FileList.csv"
    videos_dir = dataset_root / "Videos"
    tracings_csv = dataset_root / "VolumeTracings.csv"

    if not file_list_path.exists():
        return []

    traced_videos = set()
    if tracings_csv.exists():
        with tracings_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("FileName") or "").strip()
                if not name:
                    continue
                base = name[:-4] if name.lower().endswith(".avi") else name
                traced_videos.add(base)

    records: List[UnifiedRecord] = []
    with file_list_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = (row.get("FileName") or "").strip()
            if not file_name:
                continue

            base = file_name[:-4] if file_name.lower().endswith(".avi") else file_name
            video_path = videos_dir / f"{base}.avi"
            if not video_path.exists():
                alt_path = videos_dir / file_name
                if alt_path.exists():
                    video_path = alt_path

            notes: List[str] = []
            if not video_path.exists():
                notes.append("missing_video")
            if base in traced_videos:
                notes.append("has_volume_tracings")

            records.append(
                UnifiedRecord(
                    sample_id=f"echonet_{base}",
                    dataset="echonet_dynamic",
                    split=normalize_split(row.get("Split", "")),
                    task="video_ef_regression",
                    modality="avi",
                    source_image=rel_or_abs(video_path, root),
                    source_label="",
                    unified_image="",
                    unified_label="",
                    patient_id=base,
                    view="",
                    phase="full_cycle",
                    site="",
                    target_ef=(row.get("EF") or "").strip(),
                    target_esv=(row.get("ESV") or "").strip(),
                    target_edv=(row.get("EDV") or "").strip(),
                    fps=(row.get("FPS") or "").strip(),
                    num_frames=(row.get("NumberOfFrames") or "").strip(),
                    frame_height=(row.get("FrameHeight") or "").strip(),
                    frame_width=(row.get("FrameWidth") or "").strip(),
                    notes=";".join(notes),
                )
            )
    return records


def read_camus_split_map(split_dir: Path) -> Dict[str, str]:
    split_map: Dict[str, str] = {}
    file_to_split = {
        "subgroup_training.txt": "train",
        "subgroup_validation.txt": "val",
        "subgroup_testing.txt": "test",
    }
    for file_name, split in file_to_split.items():
        path = split_dir / file_name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                patient = line.strip()
                if patient:
                    split_map[patient] = split
    return split_map


def build_camus_records(root: Path) -> List[UnifiedRecord]:
    dataset_root = resolve_dataset_root(root, "CAMUS_public")
    nifti_root = dataset_root / "database_nifti"
    split_dir = dataset_root / "database_split"
    if not nifti_root.exists():
        return []

    split_map = read_camus_split_map(split_dir)
    grouped: Dict[Tuple[str, str, str], Dict[str, Optional[Path]]] = {}

    for nii in nifti_root.rglob("*.nii.gz"):
        match = CAMUS_FILE_RE.match(nii.name)
        if not match:
            continue

        patient = match.group("patient")
        view = match.group("view")
        phase = match.group("phase")
        is_gt = bool(match.group("is_gt"))
        key = (patient, view, phase)
        grouped.setdefault(key, {"image": None, "label": None})
        if is_gt:
            grouped[key]["label"] = nii
        else:
            grouped[key]["image"] = nii

    records: List[UnifiedRecord] = []
    for (patient, view, phase), paths in sorted(grouped.items()):
        image = paths.get("image")
        label = paths.get("label")

        notes: List[str] = []
        if image is None:
            notes.append("missing_image")
        if label is None:
            notes.append("missing_label")

        records.append(
            UnifiedRecord(
                sample_id=f"camus_{patient}_{view}_{phase}",
                dataset="camus",
                split=split_map.get(patient, "unknown"),
                task="segmentation",
                modality="nii.gz",
                source_image=rel_or_abs(image, root) if image else "",
                source_label=rel_or_abs(label, root) if label else "",
                unified_image="",
                unified_label="",
                patient_id=patient,
                view=view,
                phase=phase,
                site="",
                target_ef="",
                target_esv="",
                target_edv="",
                fps="",
                num_frames="",
                frame_height="",
                frame_width="",
                notes=";".join(notes),
            )
        )
    return records


def build_cardiac_udc_records(root: Path) -> List[UnifiedRecord]:
    dataset_root = resolve_dataset_root(root, "cardiacUDC_dataset")
    if not dataset_root.exists():
        return []

    records: List[UnifiedRecord] = []
    for image in sorted(dataset_root.rglob("*_image.nii.gz")):
        label = image.with_name(image.name.replace("_image.nii.gz", "_label.nii.gz"))
        has_label = label.exists()

        relative_parent = image.parent.relative_to(dataset_root)
        site = relative_parent.parts[0] if relative_parent.parts else ""

        match = CARDIAC_UDC_IMAGE_RE.match(image.name)
        if match:
            subject = match.group("subject")
            subject_id = match.group("subject_id")
            view = f"{match.group('view')}CH"
            patient_id = f"{subject}-{subject_id}"
        else:
            subject = ""
            view = ""
            patient_id = image.stem

        notes: List[str] = []
        if not has_label:
            notes.append("missing_label")

        records.append(
            UnifiedRecord(
                sample_id=f"cardiacudc_{site}_{image.name.replace('_image.nii.gz', '')}",
                dataset="cardiac_udc",
                split="unknown",
                task="segmentation",
                modality="nii.gz",
                source_image=rel_or_abs(image, root),
                source_label=rel_or_abs(label, root) if has_label else "",
                unified_image="",
                unified_label="",
                patient_id=patient_id,
                view=view,
                phase="",
                site=site,
                target_ef="",
                target_esv="",
                target_edv="",
                fps="",
                num_frames="",
                frame_height="",
                frame_width="",
                notes=";".join(notes),
            )
        )
    return records


def materialize_records(
    records: Iterable[UnifiedRecord], root: Path, output: Path, mode: str, overwrite: bool
) -> None:
    if mode == "manifest":
        return

    data_root = output / "data"
    for record in records:
        sample_dir = data_root / safe_sample_id(record.sample_id)

        if record.source_image:
            image_src = (root / record.source_image).resolve()
            if image_src.exists():
                image_dst = sample_dir / f"image{suffix_for(image_src)}"
                transfer_file(image_src, image_dst, mode, overwrite)
                record.unified_image = image_dst.resolve().relative_to(output.resolve()).as_posix()

        if record.source_label:
            label_src = (root / record.source_label).resolve()
            if label_src.exists():
                label_dst = sample_dir / f"label{suffix_for(label_src)}"
                transfer_file(label_src, label_dst, mode, overwrite)
                record.unified_label = label_dst.resolve().relative_to(output.resolve()).as_posix()


def write_manifest(records: List[UnifiedRecord], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(UnifiedRecord.__dataclass_fields__.keys())
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_summary(records: List[UnifiedRecord], summary_path: Path) -> None:
    datasets: Dict[str, int] = {}
    missing_images = 0
    missing_labels = 0

    for record in records:
        datasets[record.dataset] = datasets.get(record.dataset, 0) + 1
        if not record.source_image:
            missing_images += 1
        if record.task == "segmentation" and not record.source_label:
            missing_labels += 1

    summary = {
        "total_records": len(records),
        "dataset_counts": datasets,
        "missing_images": missing_images,
        "missing_segmentation_labels": missing_labels,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()

    records: List[UnifiedRecord] = []
    records.extend(build_echonet_records(root))
    records.extend(build_camus_records(root))
    records.extend(build_cardiac_udc_records(root))
    records.sort(key=lambda x: (x.dataset, x.sample_id))

    output.mkdir(parents=True, exist_ok=True)
    materialize_records(records, root, output, args.mode, args.overwrite)

    manifest_path = output / "manifest.csv"
    summary_path = output / "summary.json"
    write_manifest(records, manifest_path)
    write_summary(records, summary_path)

    print(f"Saved: {manifest_path}")
    print(f"Saved: {summary_path}")
    print(f"Total records: {len(records)}")


if __name__ == "__main__":
    main()
