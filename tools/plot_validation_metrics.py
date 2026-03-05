#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build grouped bar chart from validation_summary.csv produced by validate_yolo_runs.py."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("tools/runs/validation_runs/validation_summary.csv"),
        help="Path to validation summary CSV.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help="Optional run order to use in the chart.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metric columns to include.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tools/runs/validation_runs"),
        help="Directory for chart and helper CSVs.",
    )
    parser.add_argument(
        "--chart-name",
        default="validation_metrics_grouped_bar.png",
        help="Output chart PNG filename.",
    )
    parser.add_argument(
        "--title",
        default="Validation Metrics From model.val()",
        help="Chart title.",
    )
    return parser.parse_args()


def safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_summary(summary_csv: Path) -> List[Dict[str, str]]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Validation summary not found: {summary_csv.as_posix()}")
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {summary_csv.as_posix()}")
    return rows


def order_rows(rows: List[Dict[str, str]], runs: List[str]) -> List[Dict[str, str]]:
    if not runs:
        return rows

    index = {row["run"]: row for row in rows}
    missing = [run for run in runs if run not in index]
    if missing:
        available = sorted(index)
        raise FileNotFoundError(
            "Requested run(s) not found in validation summary: "
            + ", ".join(missing)
            + ". Available runs: "
            + ", ".join(available)
        )
    return [index[run] for run in runs]


def pretty_metric_label(metric: str) -> str:
    if metric.startswith("metrics/"):
        label = metric.replace("metrics/", "")
        return label.replace("(M)", "(P)")
    return metric


def write_run_mapping(rows: List[Dict[str, str]], out_dir: Path) -> Path:
    out_path = out_dir / "validation_run_versions.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "run"])
        for i, row in enumerate(rows, start=1):
            writer.writerow([f"V{i}", row["run"]])
    return out_path


def write_selected_table(rows: List[Dict[str, str]], metrics: List[str], out_dir: Path) -> Path:
    out_path = out_dir / "validation_metrics_selected.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run"] + metrics)
        for row in rows:
            writer.writerow([row["run"]] + [row.get(metric, "") for metric in metrics])
    return out_path


def plot_grouped_bars(
    rows: List[Dict[str, str]],
    metrics: List[str],
    title: str,
    out_path: Path,
) -> None:
    num_runs = len(rows)
    num_metrics = len(metrics)
    width = 0.8 / float(max(1, num_runs))
    x_positions = list(range(num_metrics))

    fig, ax = plt.subplots(figsize=(14, 8))

    for run_idx, row in enumerate(rows):
        offset = -0.4 + width * run_idx + width / 2.0
        bar_positions = [x + offset for x in x_positions]
        values: List[float] = []
        for metric in metrics:
            value = safe_float(row.get(metric, ""))
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
    summary_csv = args.summary_csv.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_summary(summary_csv)
    rows = order_rows(rows, args.runs)
    metrics = list(args.metrics)

    chart_path = out_dir / args.chart_name
    plot_grouped_bars(rows, metrics, args.title, chart_path)

    mapping_path = write_run_mapping(rows, out_dir)
    table_path = write_selected_table(rows, metrics, out_dir)

    print(f"summary={summary_csv.as_posix()}")
    print(f"runs={len(rows)}")
    for i, row in enumerate(rows, start=1):
        print(f" - V{i} -> {row['run']}")
    print(f"chart={chart_path.as_posix()}")
    print(f"mapping={mapping_path.as_posix()}")
    print(f"table={table_path.as_posix()}")


if __name__ == "__main__":
    main()
