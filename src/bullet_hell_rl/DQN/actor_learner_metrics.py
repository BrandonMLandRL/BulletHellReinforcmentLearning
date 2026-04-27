"""
Append-only CSV metrics log for actor and learner processes (cross-process safe via file lock).

Location
--------
- Environment variable ``RL_METRICS_LOG``: absolute or relative path to the CSV file.
- Default: ``./training_metrics.csv`` in the current working directory.

Format
------
- UTF-8 CSV with a header row written once when the file is created (empty or missing).
- One row per logged event. Unused columns are empty (``""``).
- Booleans are stored as ``1`` or ``0``.
- ``hidden_units`` (from RL config) is stored as a ``;``-separated list, e.g. ``128;56``.

Row kinds (``event`` column)
----------------------------
1. ``train_step`` (``source`` ``learner``): loss, replay/step counters, throughput, rewards.
2. ``step_sample`` (``source`` ``actor``): rl_step, epsilon, branch fractions, reward window,
   ``branch_count_*`` (flattened from the former ``branch_counts`` dict), etc.
3. ``life_end`` (``source`` ``actor``): ``life_episode_reward``, ``life_episode_rl_steps`` when the
   player dies (single row per death transition).
4. ``config`` (``source`` ``actor`` or ``learner``): flattened ``ActorLearnerRLConfig`` fields
   (no nested ``rl_config`` column).

Column order (all rows use this schema)
---------------------------------------
``ts``, ``ts_iso``, ``source``, ``event``,
``loss``, ``step_count``, ``replay_size``, ``min_replay_size``, ``experience_recv_total``,
``experiences_since_last_train``, ``experience_throughput_per_sec``, ``mean_reward_window``,
``reward_count_window``, ``life_episode_reward``, ``life_episode_rl_steps``,
``target_updates_pending_counter``,
``rl_step``, ``selection_index``, ``epsilon``, ``learner_connected``, ``frac_greedy``,
``frac_random``, ``frac_warmup``, ``last_action_branch``,
``branch_count_greedy``, ``branch_count_random``, ``branch_count_warmup``,
``state_dimension``, ``action_dimension``, ``hidden_units``, ``hidden_activation``, ``gamma``,
``train_freq``, ``replay_buffer_size``, ``batch_size``, ``target_network_period``,
``weights_publish_every``, ``epsilon_start``, ``number_episodes``,
``explore_pure_random_until_selection_index``, ``epsilon_decay_after_selection_index``,
``epsilon_decay_multiplier``, ``selection_index_divisor``.

Locking
-------
Uses ``<csv_path>.lock`` with ``filelock.FileLock`` so concurrent appends stay consistent.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict
from typing import Any, Mapping

from filelock import FileLock

DEFAULT_METRICS_PATH = "training_metrics.csv"

# Single source of truth for CSV header and row shape.
FIELDNAMES: tuple[str, ...] = (
    "ts",
    "ts_iso",
    "source",
    "event",
    "loss",
    "step_count",
    "replay_size",
    "min_replay_size",
    "experience_recv_total",
    "experiences_since_last_train",
    "experience_throughput_per_sec",
    "mean_reward_window",
    "reward_count_window",
    "life_episode_reward",
    "life_episode_rl_steps",
    "target_updates_pending_counter",
    "rl_step",
    "selection_index",
    "epsilon",
    "learner_connected",
    "frac_greedy",
    "frac_random",
    "frac_warmup",
    "last_action_branch",
    "branch_count_greedy",
    "branch_count_random",
    "branch_count_warmup",
    "state_dimension",
    "action_dimension",
    "hidden_units",
    "hidden_activation",
    "gamma",
    "train_freq",
    "replay_buffer_size",
    "batch_size",
    "target_network_period",
    "weights_publish_every",
    "epsilon_start",
    "number_episodes",
    "explore_pure_random_until_selection_index",
    "epsilon_decay_after_selection_index",
    "epsilon_decay_multiplier",
    "selection_index_divisor",
)

_FIELDSET = frozenset(FIELDNAMES)


def metrics_log_path() -> str:
    return os.path.abspath(os.environ.get("RL_METRICS_LOG", DEFAULT_METRICS_PATH))


def _cell_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    return str(value)


def _expand_branch_counts(merged: dict[str, Any]) -> None:
    raw = merged.pop("branch_counts", None)
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise TypeError(f"branch_counts must be a dict, got {type(raw).__name__}")
    merged["branch_count_greedy"] = raw.get("greedy", 0)
    merged["branch_count_random"] = raw.get("random", 0)
    merged["branch_count_warmup"] = raw.get("warmup", 0)


def _row_from_merged(merged: dict[str, Any]) -> dict[str, str]:
    row = {k: "" for k in FIELDNAMES}
    for key, val in merged.items():
        if key not in _FIELDSET:
            raise ValueError(f"Unknown metrics field: {key!r}")
        row[key] = _cell_str(val)
    return row


def log_metrics_record(source: str, event: str, fields: Mapping[str, Any]) -> None:
    path = metrics_log_path()
    lock_path = path + ".lock"
    merged: dict[str, Any] = {
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "source": source,
        "event": event,
    }
    merged.update(dict(fields))
    _expand_branch_counts(merged)
    row = _row_from_merged(merged)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    lock = FileLock(lock_path, timeout=15)
    with lock:
        need_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(FIELDNAMES),
                extrasaction="raise",
            )
            if need_header:
                writer.writeheader()
            writer.writerow(row)


def log_rl_config_snapshot(source: str, cfg: Any) -> None:
    """Log a frozen snapshot of ActorLearnerRLConfig (or any compatible dataclass)."""
    flat = asdict(cfg)
    hu = flat.get("hidden_units")
    if isinstance(hu, (list, tuple)):
        flat["hidden_units"] = ";".join(str(x) for x in hu)
    log_metrics_record(source, "config", flat)


def actor_metrics_log_interval() -> int:
    return max(1, int(os.environ.get("RL_METRICS_ACTOR_EVERY", "50")))
