"""
Shared RL bridge: map multiplayer server JSON updates to DQN observations.

Observation layout (flat float32, length STATE_DIM from actor_learner_rl_config):
  [8 density cones | 5×4 hostile bullets | 2×4 allies | 3×4 enemies | 1 health]

Each entity block uses obs_transform (rel_x, rel_y, rel_v_x, rel_v_y) with positions
scaled by world size and relative velocities normalized per entity type.
"""
from __future__ import annotations

import math
from platform import python_revision
from typing import Any, Iterable, Mapping, Sequence, TypedDict

import numpy as np

from bullet_hell_rl.bullethell import (
    BULLET_SPEED_ENEMY,
    ENEMY_SPEED,
    PLAYER_HEALTH_MAX,
    PLAYER_SPEED,
    WORLD_HEIGHT,
    WORLD_WIDTH,
)
from bullet_hell_rl.DQN.actor_learner_rl_config import ACTOR_LEARNER_RL_CONFIG

STATE_DIM = ACTOR_LEARNER_RL_CONFIG.state_dimension
ACTION_DIM = ACTOR_LEARNER_RL_CONFIG.action_dimension
N_DENSITY_CONES = 8
N_BULLETS = 5
N_ALLIES = 2
N_ENEMIES = 3
OBS_TRANSFORM_DIM = 4
N_HEALTH_FEATURES = 1

CONE_RADIUS = .1 * WORLD_HEIGHT

_EXPECTED_DIM = (
    N_DENSITY_CONES
    + N_BULLETS * OBS_TRANSFORM_DIM
    + N_ALLIES * OBS_TRANSFORM_DIM
    + N_ENEMIES * OBS_TRANSFORM_DIM
    + N_HEALTH_FEATURES
)
assert _EXPECTED_DIM == STATE_DIM, (_EXPECTED_DIM, STATE_DIM)

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


def _nearest_entities(
    ref_x: float,
    ref_y: float,
    entities: Iterable[Mapping[str, Any]],
    limit: int,
) -> list[Mapping[str, Any]]:
    entities = list(entities)
    if not entities:
        return []

    def dist2(e: Mapping[str, Any]) -> float:
        ex = float(e.get("x", 0.0))
        ey = float(e.get("y", 0.0))
        return (ex - ref_x) ** 2 + (ey - ref_y) ** 2

    entities.sort(key=dist2)
    return entities[:limit]


def _player_velocity_px(you: Mapping[str, Any]) -> tuple[float, float]:
    """Entity-style vel_x/vel_y in {-1,0,1} times PLAYER_SPEED → px/s."""
    vx = float(you.get("vel_x", 0.0))
    vy = float(you.get("vel_y", 0.0))
    return vx, vy


def _density_cone_features(
    bullets,
    px: float,
    py: float,
) -> np.ndarray:
    """Hostile bullet counts in 8 angular wedges within CONE_RADIUS, L1-normalized."""
    # Bullet format example:
    # {'x', 'y', 'vel_x', 'vel_y', 'is_friendly', 'size', 'owner_id'}

    def get_bullets_in_cone_radius(px: float, py: float, r: float):
        blts: list[Any] = []
        p = (px, py)
        for b in bullets:
            if b.get("is_friendly", False):
                continue
            distance = math.dist(p, (b.get("x"), b.get("y")))
            if distance <= r:
                blts.append(b)
        return blts

    bullets_in_radius = get_bullets_in_cone_radius(px, py, CONE_RADIUS)

    buckets = [0.0] * N_DENSITY_CONES
    for b in bullets_in_radius:
        dx = float(b.get("x", 0.0)) - px
        dy = float(b.get("y", 0.0)) - py
        rad = np.arctan2(dy, dx)
        bkt = int(np.floor((rad + np.pi) / (np.pi / 4))) % N_DENSITY_CONES
        buckets[bkt] += 1.0

    n = len(bullets_in_radius)
    if n > 0:
        for i in range(N_DENSITY_CONES):
            buckets[i] /= n

    return np.asarray(buckets, dtype=np.float32)


def build_obs_from_update(
    update_msg: UpdateMessage,
    prev_update_msg: UpdateMessage | None = None,
) -> np.ndarray:
    """
    Build a fixed-length flat observation from MSG_UPDATE.
    prev_update_msg is accepted for API compatibility; unused.
    """
    _ = prev_update_msg
    you = update_msg.get("you") or {}
    px = float(you.get("x", 0.0))
    py = float(you.get("y", 0.0))
    health = float(you.get("health", 0.0))
    pvx, pvy = _player_velocity_px(you)

    bullets_raw = list(update_msg.get("bullets") or [])

    cones = _density_cone_features(bullets_raw, px, py)

    hostile = [b for b in bullets_raw if not b.get("is_friendly", False)]
    nearest_bu = _nearest_entities(px, py, hostile, N_BULLETS)
    bullets_obs = np.zeros((N_BULLETS, OBS_TRANSFORM_DIM), dtype=np.float32)
    for i, b in enumerate(nearest_bu[:N_BULLETS]):
        bvx = float(b.get("vel_x", 0.0))
        bvy = float(b.get("vel_y", 0.0))
        bullets_obs[i] = np.array(
            [
                (float(b.get("x", 0.0)) - px) / WORLD_WIDTH,
                (float(b.get("y", 0.0)) - py) / WORLD_HEIGHT,
                (bvx - pvx) / BULLET_SPEED_ENEMY,
                (bvy - pvy) / BULLET_SPEED_ENEMY,
            ],
            dtype=np.float32,
        )

    allies_raw = [
        p
        for p in (update_msg.get("players") or [])
        if float(p.get("health", 0.0)) > 0.0
    ]
    nearest_al = _nearest_entities(px, py, allies_raw, N_ALLIES)
    allies_obs = np.zeros((N_ALLIES, OBS_TRANSFORM_DIM), dtype=np.float32)
    for i, ally in enumerate(nearest_al[:N_ALLIES]):
        avx, avy = _player_velocity_px(ally)
        allies_obs[i] = np.array(
            [
                (float(ally.get("x", 0.0)) - px) / WORLD_WIDTH,
                (float(ally.get("y", 0.0)) - py) / WORLD_HEIGHT,
                (avx - pvx) / PLAYER_SPEED,
                (avy - pvy) / PLAYER_SPEED,
            ],
            dtype=np.float32,
        )

    enemies_raw = list(update_msg.get("enemies") or [])
    nearest_en = _nearest_entities(px, py, enemies_raw, N_ENEMIES)
    enemies_obs = np.zeros((N_ENEMIES, OBS_TRANSFORM_DIM), dtype=np.float32)
    for i, enemy in enumerate(nearest_en[:N_ENEMIES]):
        evx = float(enemy.get("vel_x", 0.0)) * ENEMY_SPEED
        evy = float(enemy.get("vel_y", 0.0)) * ENEMY_SPEED
        # Relative enemy motion vs player, normalized by ENEMY_SPEED.
        enemies_obs[i] = np.array(
            [
                (float(enemy.get("x", 0.0)) - px) / WORLD_WIDTH,
                (float(enemy.get("y", 0.0)) - py) / WORLD_HEIGHT,
                (evx - pvx) / ENEMY_SPEED,
                (evy - pvy) / ENEMY_SPEED,
            ],
            dtype=np.float32,
        )

    health_feat = np.array([health / PLAYER_HEALTH_MAX], dtype=np.float32)

    flat = np.concatenate(
        [
            cones,
            bullets_obs.ravel(),
            allies_obs.ravel(),
            enemies_obs.ravel(),
            health_feat,
        ],
        axis=0,
    )
    assert flat.shape[0] == STATE_DIM, flat.shape
    return flat.astype(np.float32, copy=False)


def _is_player_in_center(px: float, py: float) -> bool:
    c_x = WORLD_WIDTH / 2
    c_y = WORLD_HEIGHT / 2
    return (px > c_x / 2 and px < c_x / 2 + c_x and py > c_y / 2 and py < c_y / 2 + c_y)


def compute_reward_and_done(
    prev_update_msg: UpdateMessage | None,
    curr_update_msg: UpdateMessage,
    tick_delta: int | float | None,
) -> tuple[float, bool, MetaDict]:
    """
    Reward aligned with BulletHellEnv priorities: kill bonus, damage penalty, center bonus, else +1.
    """
    _ = tick_delta
    meta: MetaDict = {}
    if curr_update_msg.get("type") != "update":
        return 0.0, False, meta

    you_curr = curr_update_msg.get("you") or {}
    h_curr = float(you_curr.get("health", 0.0))
    k_curr = int(you_curr.get("kill_count", 0))
    px = float(you_curr.get("x", 0.0))
    py = float(you_curr.get("y", 0.0))

    if prev_update_msg is None or prev_update_msg.get("type") != "update":
        return 1.0, False, meta

    you_prev = prev_update_msg.get("you") or {}
    h_prev = float(you_prev.get("health", 0.0))
    k_prev = int(you_prev.get("kill_count", 0))

    if k_curr > k_prev:
        dk = k_curr - k_prev
        reward = 100.0 * dk
        meta["reason"] = "kill"
    elif h_curr < h_prev:
        reward = -300.0
        meta["reason"] = "damage"
    elif _is_player_in_center(px, py):
        reward = 10.0
        meta["reason"] = "center"
    else:
        reward = 1.0
        meta["reason"] = "safe"

    done = h_curr <= 0
    if done:
        meta["reason"] = "death"
    return reward, done, meta


def serialize_experience(
    state: np.ndarray | Sequence[float],
    action: int,
    reward: float,
    next_state: np.ndarray | Sequence[float],
    done: bool,
    meta: Mapping[str, Any] | None = None,
) -> ExperienceTuple:
    state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
    next_state_arr = np.asarray(next_state, dtype=np.float32).reshape(-1)
    return {
        "state": state_arr.tolist(),
        "action": int(action),
        "reward": float(reward),
        "next_state": next_state_arr.tolist(),
        "done": bool(done),
        "meta": dict(meta or {}),
    }


def validate_experience_shape(exp: Mapping[str, Any]) -> None:
    for key in ("state", "action", "reward", "next_state", "done"):
        if key not in exp:
            raise ValueError(f"experience missing key: {key}")

    state = exp["state"]
    next_state = exp["next_state"]
    if not isinstance(state, (list, tuple)) or len(state) != STATE_DIM:
        raise ValueError(f"state must have length {STATE_DIM}, got {state!r}")
    if not isinstance(next_state, (list, tuple)) or len(next_state) != STATE_DIM:
        raise ValueError(f"next_state must have length {STATE_DIM}, got {next_state!r}")

    action = int(exp["action"])
    if not (0 <= action < ACTION_DIM):
        raise ValueError(f"action must be in [0, {ACTION_DIM}), got {action}")

    if not math.isfinite(float(exp["reward"])):
        raise ValueError("reward must be finite")

    for name, vec in ("state", state), ("next_state", next_state):
        for i, v in enumerate(vec):
            if not math.isfinite(float(v)):
                raise ValueError(f"{name}[{i}] is not finite: {v}")


def run_bridge_self_checks() -> None:
    dummy_update = {
        "type": "update",
        "you": {
            "id": 0,
            "x": 100.0,
            "y": 200.0,
            "health": 100.0,
            "kill_count": 0,
            "size": 20,
            "vel_x": 1.0,
            "vel_y": 0.0,
        },
        "players": [
            {
                "id": 1,
                "x": 300.0,
                "y": 200.0,
                "health": 80.0,
                "kill_count": 0,
                "size": 20,
                "vel_x": 0.0,
                "vel_y": -1.0,
            },
        ],
        "enemies": [
            {"x": 150.0, "y": 200.0, "vel_x": 1.0, "vel_y": 0.0, "size": 20},
        ],
        "bullets": [
            {"x": 120.0, "y": 210.0, "vel_x": 50.0, "vel_y": 0.0, "is_friendly": False, "size": 10},
        ],
        "tick": 1,
    }
    obs = build_obs_from_update(dummy_update)
    assert obs.shape == (STATE_DIM,)
    r, d, _ = compute_reward_and_done(None, dummy_update, None)
    assert r == 1.0 and not d
    exp = serialize_experience(obs, 3, 1.0, obs, False)
    validate_experience_shape(exp)


__all__ = [
    "STATE_DIM",
    "ACTION_DIM",
    "N_DENSITY_CONES",
    "N_BULLETS",
    "N_ALLIES",
    "N_ENEMIES",
    "OBS_TRANSFORM_DIM",
    "N_HEALTH_FEATURES",
    "ExperienceTuple",
    "build_obs_from_update",
    "compute_reward_and_done",
    "serialize_experience",
    "validate_experience_shape",
    "run_bridge_self_checks",
]
