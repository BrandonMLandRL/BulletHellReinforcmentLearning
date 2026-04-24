"""
Shared JSONL metrics log for actor and learner processes (cross-process safe via file lock).

Path: env RL_METRICS_LOG or ./training_metrics.jsonl
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Mapping

from filelock import FileLock

DEFAULT_METRICS_PATH = "training_metrics.jsonl"


def metrics_log_path() -> str:
    return os.path.abspath(os.environ.get("RL_METRICS_LOG", DEFAULT_METRICS_PATH))


def log_metrics_record(source: str, event: str, fields: Mapping[str, Any]) -> None:
    path = metrics_log_path()
    lock_path = path + ".lock"
    rec: dict[str, Any] = {
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "source": source,
        "event": event,
    }
    rec.update(dict(fields))
    line = json.dumps(rec, default=str) + "\n"
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    lock = FileLock(lock_path, timeout=15)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def log_rl_config_snapshot(source: str, cfg: Any) -> None:
    """Log a frozen snapshot of ActorLearnerRLConfig (or any dataclass)."""
    log_metrics_record(source, "config", {"rl_config": asdict(cfg)})


def actor_metrics_log_interval() -> int:
    return max(1, int(os.environ.get("RL_METRICS_ACTOR_EVERY", "50")))
