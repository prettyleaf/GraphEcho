#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METRICS = [
    "metrics/mAP50-95(B)",
    "metrics/mAP50(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50-95(M)",
    "metrics/mAP50(M)",
    "metrics/precision(M)",
    "metrics/recall(M)",
]

PURPLE_PALETTE = [
    "#B9A0FF",
    "#9D6CFF",
    "#7E3FF2",
    "#5A20B8",
    "#330066",
]


@dataclass
class RunMetrics:
    name: str
    csv_path: Path
    columns: Dict[str, List[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build grouped comparison bar chart for YOLO validation metrics."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("tools/runs"),
        help="Directory with run subfolders containing results.csv.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help="Optional specific run folder names.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tools/runs/metric_plots"),
        help="Output directory for chart and CSV tables.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metric columns to put on X axis.",
    )
    parser.add_argument(
        "--epoch-policy",
        choices=["best", "last"],
        default="best",
        help="How to pick epoch values from each run.",
    )
    parser.add_argument(
        "--select-metric",
        default="metrics/mAP50-95(M)",
        help="Metric used to choose best epoch when --epoch-policy=best.",
    )
    parser.add_argument(
        "--title",
        default="Validation Metrics Comparison",
        help="Chart title.",
    )
    parser.add_argument(
        "--chart-name",
        default="val_metrics_grouped_bar.png",
        help="Output chart file name.",
    )
    return parser.parse_args()


def safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def discover_run_dirs(runs_dir: Path, selected: Iterable[str]) -> List[Path]:
    selected_list = list(selected)
    out: List[Path] = []

    if selected_list:
        missing: List[str] = []
        for run_name in selected_list:
            run_dir = runs_dir / run_name
            if not run_dir.is_dir() or not (run_dir / "results.csv").exists():
                missing.append(run_name)
                continue
            out.append(run_dir)
        if missing:
            available = sorted(
                d.name for d in runs_dir.iterdir() if d.is_dir() and (d / "results.csv").exists()
            )
            raise FileNotFoundError(
                "Requested run(s) not found: "
                + ", ".join(missing)
                + ". Available runs: "
                + ", ".join(available)
            )
        return out

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if (run_dir / "results.csv").exists():
            out.append(run_dir)
    return out


def read_run_metrics(run_dir: Path) -> RunMetrics:
    csv_path = run_dir / "results.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"Empty header in {csv_path.as_posix()}")

        fieldnames = [name.strip() for name in reader.fieldnames]
        columns: Dict[str, List[float]] = {name: [] for name in fieldnames}

        for row in reader:
            normalized = {k.strip(): v for k, v in row.items()}
            for name in fieldnames:
                value = safe_float(normalized.get(name, ""))
                if value is not None:
                    columns[name].append(value)

    if "epoch" not in columns or not columns["epoch"]:
        raise RuntimeError(f"Invalid epochs in {csv_path.as_posix()}")
    return RunMetrics(name=run_dir.name, csv_path=csv_path, columns=columns)


def select_epoch_index(run: RunMetrics, epoch_policy: str, select_metric: str) -> int:
    epochs = run.columns.get("epoch", [])
    if not epochs:
        return 0
    if epoch_policy == "last":
        return len(epochs) - 1

    selector = run.columns.get(select_metric, [])
    n = min(len(epochs), len(selector))
    if n == 0:
        return len(epochs) - 1
    return max(range(n), key=lambda i: selector[i])


def metric_value_at(run: RunMetrics, metric: str, idx: int) -> Optional[float]:
    values = run.columns.get(metric, [])
    if idx < 0 or idx >= len(values):
        return None
    return values[idx]


def pretty_metric_label(metric: str) -> str:
    if metric.startswith("metrics/"):
        label = metric.replace("metrics/", "")
        return label.replace("(M)", "(P)")
    return metric


def write_metric_table(
    runs: List[RunMetrics],
    metrics: List[str],
    epoch_indices: Dict[str, int],
    out_dir: Path,
) -> Path:
    out_path = out_dir / "val_metrics_selected_epoch.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "epoch"] + metrics + ["results_csv"])
        for run in runs:
            idx = epoch_indices[run.name]
            epoch_value = int(run.columns["epoch"][idx]) if idx < len(run.columns["epoch"]) else ""
            row = [run.name, epoch_value]
            for metric in metrics:
                value = metric_value_at(run, metric, idx)
                row.append("" if value is None else f"{value:.6f}")
            row.append(run.csv_path.as_posix())
            writer.writerow(row)
    return out_path


def write_run_mapping(runs: List[RunMetrics], out_dir: Path) -> Path:
    out_path = out_dir / "run_versions.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "run"])
        for i, run in enumerate(runs, start=1):
            writer.writerow([f"V{i}", run.name])
    return out_path


def plot_grouped_bars(
    runs: List[RunMetrics],
    metrics: List[str],
    epoch_indices: Dict[str, int],
    title: str,
    out_path: Path,
) -> None:
    num_runs = len(runs)
    num_metrics = len(metrics)
    width = 0.8 / float(max(1, num_runs))
    x_positions = list(range(num_metrics))

    fig, ax = plt.subplots(figsize=(14, 8))

    for run_idx, run in enumerate(runs):
        offset = -0.4 + width * run_idx + width / 2.0
        bar_positions = [x + offset for x in x_positions]
        values: List[float] = []
        idx = epoch_indices[run.name]
        for metric in metrics:
            value = metric_value_at(run, metric, idx)
            values.append(0.0 if value is None else value)

        color = PURPLE_PALETTE[run_idx % len(PURPLE_PALETTE)]
        bars = ax.bar(
            bar_positions,
            values,
            width=width,
            label=f"V{run_idx + 1}",
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
        )

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"),
                color="white",
            )

    pretty_labels = [pretty_metric_label(m) for m in metrics]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pretty_labels, rotation=35, ha="right", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=14)
    ax.set_ylim(0.6, 1.2)
    ax.set_title(title, fontsize=16, weight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_axisbelow(True)

    legend = ax.legend(title="Model Versions", loc="upper left", framealpha=0.95)
    legend.get_title().set_fontsize(12)
    for t in legend.get_texts():
        t.set_fontsize(11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir.as_posix()}")

    run_dirs = discover_run_dirs(runs_dir, args.runs)
    if not run_dirs:
        raise RuntimeError(f"No run folders with results.csv in: {runs_dir.as_posix()}")

    runs = [read_run_metrics(d) for d in run_dirs]
    metrics = list(args.metrics)

    epoch_indices: Dict[str, int] = {}
    for run in runs:
        epoch_indices[run.name] = select_epoch_index(run, args.epoch_policy, args.select_metric)

    chart_path = out_dir / args.chart_name
    plot_grouped_bars(
        runs=runs,
        metrics=metrics,
        epoch_indices=epoch_indices,
        title=args.title,
        out_path=chart_path,
    )

    table_path = write_metric_table(runs, metrics, epoch_indices, out_dir)
    mapping_path = write_run_mapping(runs, out_dir)

    print(f"runs={len(runs)}")
    for i, run in enumerate(runs, start=1):
        idx = epoch_indices[run.name]
        epoch_value = int(run.columns['epoch'][idx]) if idx < len(run.columns['epoch']) else "?"
        print(f" - V{i} -> {run.name}, epoch={epoch_value}, csv={run.csv_path.as_posix()}")
    print(f"epoch_policy={args.epoch_policy}")
    print(f"select_metric={args.select_metric}")
    print(f"chart={chart_path.as_posix()}")
    print(f"table={table_path.as_posix()}")
    print(f"mapping={mapping_path.as_posix()}")


if __name__ == "__main__":
    main()
