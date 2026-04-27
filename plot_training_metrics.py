#!/usr/bin/env python3
"""
Stream training_metrics CSV (or legacy JSONL) and write one PNG per numeric metric.

Usage:
  python plot_training_metrics.py
  python plot_training_metrics.py path/to/metrics.csv --out-dir metrics_plots
  python plot_training_metrics.py legacy.jsonl --out-dir metrics_plots
  python plot_training_metrics.py --stride 10 --max-lines 100000  # cap data lines read

Default input path: env RL_METRICS_LOG if set, else ./training_metrics.csv
(mirrors bullet_hell_rl.DQN.actor_learner_metrics.metrics_log_path).

If the path ends with .jsonl, each line is parsed as JSON (legacy format).
Config rows (event=config) are skipped for plotting. X-axis defaults to wall-clock ts;
use --x to use another numeric column (rows with empty values are skipped).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt

DEFAULT_METRICS_PATH = "training_metrics.csv"

META_KEYS = frozenset({"ts", "ts_iso", "source", "event"})
# Legacy JSONL only; CSV has no nested keys.
SKIP_VALUE_KEYS_JSONL = frozenset({"rl_config", "branch_counts"})


def default_input_path() -> str:
    return os.path.abspath(os.environ.get("RL_METRICS_LOG", DEFAULT_METRICS_PATH))


def coerce_plottable(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def _process_record(
    rec: dict[str, Any],
    *,
    series: dict[str, list[tuple[float, float, str]]],
    x_key: str,
    stride: int,
    plottable_idx: int,
    legacy_jsonl: bool,
) -> int:
    if rec.get("event") == "config":
        return plottable_idx

    next_idx = plottable_idx + 1
    if (next_idx - 1) % stride != 0:
        return next_idx

    src = str(rec.get("source", "unknown"))
    x_val = coerce_plottable(rec.get(x_key))
    if x_val is None:
        return next_idx

    skip_vals = SKIP_VALUE_KEYS_JSONL if legacy_jsonl else frozenset()

    for key, val in rec.items():
        if key is None:
            continue
        if key in META_KEYS or key in skip_vals or key == x_key:
            continue
        if isinstance(val, (dict, list)):
            continue
        y = coerce_plottable(val)
        if y is None:
            continue
        series[key].append((x_val, y, src))

    return next_idx


def collect_series(
    path: str,
    x_key: str,
    stride: int,
    max_lines: int | None,
) -> dict[str, list[tuple[float, float, str]]]:
    """field_name -> list of (x, y, source)."""
    series: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    plottable_idx = 0
    stride = max(1, stride)
    non_empty_lines = 0
    legacy_jsonl = path.lower().endswith(".jsonl")

    if legacy_jsonl:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                non_empty_lines += 1
                if max_lines is not None and non_empty_lines > max_lines:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                plottable_idx = _process_record(
                    rec,
                    series=series,
                    x_key=x_key,
                    stride=stride,
                    plottable_idx=plottable_idx,
                    legacy_jsonl=True,
                )
    else:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for rec in reader:
                non_empty_lines += 1
                if max_lines is not None and non_empty_lines > max_lines:
                    break
                plottable_idx = _process_record(
                    rec,
                    series=series,
                    x_key=x_key,
                    stride=stride,
                    plottable_idx=plottable_idx,
                    legacy_jsonl=False,
                )

    return series


def plot_all(
    series: dict[str, list[tuple[float, float, str]]],
    out_dir: str,
    x_key: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for field, points in sorted(series.items()):
        if len(points) < 2:
            continue
        by_source: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for x, y, src in points:
            by_source[src].append((x, y))
        fig, ax = plt.subplots(figsize=(10, 4))
        for src in sorted(by_source.keys()):
            pts = sorted(by_source[src], key=lambda t: t[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, label=src, linewidth=0.8)
        ax.set_xlabel(x_key)
        ax.set_ylabel(field)
        ax.set_title(field)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{sanitize_filename(field)}.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot numeric metrics from training_metrics CSV or legacy JSONL (streaming)."
    )
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help=f"CSV or .jsonl path (default: {DEFAULT_METRICS_PATH} or RL_METRICS_LOG)",
    )
    p.add_argument(
        "--out-dir",
        default="metrics_plots",
        help="Directory for PNG files (default: metrics_plots)",
    )
    p.add_argument(
        "--x",
        dest="x_key",
        default="ts",
        help="X-axis field name (default: ts)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth plottable row (non-config) (default: 1 = all)",
    )
    p.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Stop after this many data rows (CSV rows or JSONL lines) (default: no limit)",
    )
    args = p.parse_args()
    path = os.path.abspath(args.input) if args.input else default_input_path()
    if not os.path.isfile(path):
        raise SystemExit(f"Input file not found: {path}")

    series = collect_series(path, args.x_key, args.stride, args.max_lines)
    if not series:
        raise SystemExit("No plottable series found (check --x and file contents).")
    plot_all(series, os.path.abspath(args.out_dir), args.x_key)
    n_plots = sum(1 for pts in series.values() if len(pts) >= 2)
    print(f"Wrote {n_plots} figure(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
