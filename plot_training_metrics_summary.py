#!/usr/bin/env python3
"""
Plot a single 2×2 summary figure: per-life reward, epsilon, loss, actor mean reward
window vs time.

Uses the same CSV / JSONL input and streaming logic as plot_training_metrics.py.

Usage:
  python plot_training_metrics_summary.py
  python plot_training_metrics_summary.py path/to/metrics.csv --out metrics_plots/training_summary.png
  python plot_training_metrics_summary.py --stride 5 --max-lines 50000

Default input: env RL_METRICS_LOG if set, else ./training_metrics.csv
(mirrors bullet_hell_rl.DQN.actor_learner_metrics.metrics_log_path).
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from plot_training_metrics import collect_series, default_input_path

SUMMARY_FIELDS: tuple[str, ...] = (
    "life_episode_reward",
    "epsilon",
    "loss",
    "mean_reward_window",
)


def _plot_sources_on_ax(
    ax,
    points: list[tuple[float, float, str]],
    *,
    x_key: str,
    ylabel: str,
    title: str,
    marker: str | None = None,
    add_best_fit: bool = False,
) -> None:
    if len(points) < 2:
        ax.text(
            0.5,
            0.5,
            "insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
        )
        ax.set_title(title)
        return
    by_source: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for x, y, src in points:
        by_source[src].append((x, y))
    for src in sorted(by_source.keys()):
        pts = sorted(by_source[src], key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plot_kw: dict = {"label": src, "linewidth": 0.8}
        if marker:
            plot_kw["marker"] = marker
            plot_kw["markersize"] = 4
        (line,) = ax.plot(xs, ys, **plot_kw)
        if add_best_fit and len(xs) >= 2:
            n = float(len(xs))
            sum_x = sum(xs)
            sum_y = sum(ys)
            sum_xx = sum(x * x for x in xs)
            sum_xy = sum(x * y for x, y in zip(xs, ys))
            denom = (n * sum_xx) - (sum_x * sum_x)
            if denom != 0.0:
                slope = ((n * sum_xy) - (sum_x * sum_y)) / denom
                intercept = (sum_y - (slope * sum_x)) / n
                fit_ys = [(slope * x) + intercept for x in xs]
                ax.plot(
                    xs,
                    fit_ys,
                    linestyle="--",
                    linewidth=1.2,
                    color="red",
                    label=f"{src} best fit",
                )
    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def main() -> None:
    p = argparse.ArgumentParser(
        description="2×2 summary plot from training_metrics CSV or legacy JSONL."
    )
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="CSV or .jsonl path (default: training_metrics.csv or RL_METRICS_LOG)",
    )
    p.add_argument(
        "--out",
        default=os.path.join("metrics_plots", "training_summary.png"),
        help="Output PNG path (default: metrics_plots/training_summary.png)",
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
        help="Stop after this many data rows (default: no limit)",
    )
    args = p.parse_args()
    path = os.path.abspath(args.input) if args.input else default_input_path()
    if not os.path.isfile(path):
        raise SystemExit(f"Input file not found: {path}")

    series = collect_series(path, args.x_key, args.stride, args.max_lines)
    if not series:
        raise SystemExit("No plottable series found (check --x and file contents).")

    lep = series.get("life_episode_reward", [])
    if len(lep) < 1:
        print(
            "Note: no life_episode_reward in this file. Per-life metrics need a CSV "
            "written with the current actor_learner_metrics schema (or a new metrics file)."
        )

    mrw_actor = [
        (x, y, s)
        for x, y, s in series.get("mean_reward_window", [])
        if s == "actor"
    ]
    if len(mrw_actor) < 1:
        print(
            "Note: no mean_reward_window rows with source=actor in this file. "
            "Actor step_sample metrics need the current actor_learner_metrics schema."
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, field in zip(axes.flat, SUMMARY_FIELDS):
        pts = series.get(field, [])
        if field == "mean_reward_window":
            pts = [(x, y, s) for x, y, s in pts if s == "actor"]
        mark = "o" if field == "life_episode_reward" else None
        add_fit = field == "life_episode_reward"
        _plot_sources_on_ax(
            ax,
            pts,
            x_key=args.x_key,
            ylabel=field,
            title=field,
            marker=mark,
            add_best_fit=add_fit,
        )

    fig.tight_layout()
    out_abs = os.path.abspath(args.out)
    parent = os.path.dirname(out_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_abs, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_abs}")


if __name__ == "__main__":
    main()
