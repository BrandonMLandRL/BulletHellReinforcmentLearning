"""
Shared RL bridge skeleton for actor/learner integration.

This module intentionally defines only contracts and stubs for now.
Behavioral logic is left for a follow-up implementation pass.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, TypedDict

import numpy as np

# Canonical DQN dimensions (must match DQNLegacy assumptions).
STATE_DIM = 63
ACTION_DIM = 20
N_ENEMIES = 5
N_BULLETS = 10
PLAYER_FEATURES = 3
ENTITY_FEATURES = 4


UpdateMessage = Mapping[str, Any]
MetaDict = dict[str, Any]


class ExperienceTuple(TypedDict):
    """JSON-serializable replay tuple passed from actor to learner."""

    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool
    meta: MetaDict


def build_obs_from_update(
    update_msg: UpdateMessage,
    prev_update_msg: UpdateMessage | None = None,
) -> np.ndarray:
    """
    Build a fixed-length flat observation vector from a server update packet.

    Expected vector layout (length = STATE_DIM):
    - player: 3 features
    - enemies: N_ENEMIES * 4 features
    - bullets: N_BULLETS * 4 features

    TODO (implementation phase):
    - Extract the local player anchor from update payload.
    - Select nearest enemies and hostile bullets.
    - Normalize and clip feature ranges consistently.
    - Pad missing entities with zeros.
    - Flatten in canonical order and return dtype float32.
    """
    _ = (update_msg, prev_update_msg)
    raise NotImplementedError("build_obs_from_update is a skeleton stub.")


def compute_reward_and_done(
    prev_update_msg: UpdateMessage | None,
    curr_update_msg: UpdateMessage,
    tick_delta: int | float | None,
) -> tuple[float, bool, MetaDict]:
    """
    Compute reward and terminal flag from sequential multiplayer updates.

    Returns:
    - reward: scalar float
    - done: episode terminal flag for replay tuple semantics
    - meta: supplemental details used for debugging/auditing

    TODO (implementation phase):
    - Detect health-loss transitions and death/respawn semantics.
    - Apply kill-count delta bonus when available.
    - Include deterministic reason codes in meta.
    """
    _ = (prev_update_msg, curr_update_msg, tick_delta)
    raise NotImplementedError("compute_reward_and_done is a skeleton stub.")


def serialize_experience(
    state: np.ndarray | Sequence[float],
    action: int,
    reward: float,
    next_state: np.ndarray | Sequence[float],
    done: bool,
    meta: Mapping[str, Any] | None = None,
) -> ExperienceTuple:
    """
    Serialize transition values into network-safe experience tuple format.

    Note: this skeleton currently performs only basic shape normalization and
    delegates strict checks to `validate_experience_shape`.
    """
    state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
    next_state_arr = np.asarray(next_state, dtype=np.float32).reshape(-1)
    exp: ExperienceTuple = {
        "state": state_arr.tolist(),
        "action": int(action),
        "reward": float(reward),
        "next_state": next_state_arr.tolist(),
        "done": bool(done),
        "meta": dict(meta or {}),
    }
    return exp


def validate_experience_shape(exp: Mapping[str, Any]) -> None:
    """
    Validate serialized transition structure and bounds.

    TODO (implementation phase):
    - Require all fields and expected types.
    - Enforce state lengths equal STATE_DIM.
    - Enforce action bounds [0, ACTION_DIM).
    - Enforce finite numeric values across vectors/reward.
    - Raise ValueError with clear failure reasons.
    """
    _ = exp
    raise NotImplementedError("validate_experience_shape is a skeleton stub.")


def _normalize_unit(value: float, denom: float) -> float:
    """Optional helper stub for [0,1] normalization by denominator."""
    _ = (value, denom)
    raise NotImplementedError("_normalize_unit is a skeleton helper stub.")


def _nearest_entities(
    ref_x: float,
    ref_y: float,
    entities: Iterable[Mapping[str, Any]],
    limit: int,
) -> list[Mapping[str, Any]]:
    """Optional helper stub for nearest-entity selection by squared distance."""
    _ = (ref_x, ref_y, entities, limit)
    raise NotImplementedError("_nearest_entities is a skeleton helper stub.")


def run_bridge_self_checks() -> None:
    """
    Optional skeleton hook for deterministic bridge checks.

    Intentionally left as a stub in baseline phase.
    """
    raise NotImplementedError("run_bridge_self_checks is a skeleton stub.")


__all__ = [
    "STATE_DIM",
    "ACTION_DIM",
    "N_ENEMIES",
    "N_BULLETS",
    "PLAYER_FEATURES",
    "ENTITY_FEATURES",
    "ExperienceTuple",
    "build_obs_from_update",
    "compute_reward_and_done",
    "serialize_experience",
    "validate_experience_shape",
    "run_bridge_self_checks",
]
