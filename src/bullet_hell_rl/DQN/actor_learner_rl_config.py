"""
Single source of truth for actor–learner DQN: dimensions, network, training, exploration.

Changing state_dimension requires a matching build_obs_from_update in actor_learner_rl_bridge
(and server payloads). action_dimension must match net.protocol.FLAT_ACTION_COUNT.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActorLearnerRLConfig:
    state_dimension: int = 49
    action_dimension: int = 20
    hidden_units: tuple[int, ...] = (128, 56)
    hidden_activation: str = "relu"

    gamma: float = 0.99
    train_freq: int = 4
    replay_buffer_size: int = 20000
    batch_size: int = 64
    min_replay_size: int = 2000
    target_network_period: int = 1000
    weights_publish_every: int = 100

    epsilon_start: float = 1.0
    number_episodes: int = 10**9

    explore_pure_random_until_selection_index: int = 1
    epsilon_decay_after_selection_index: int = 200
    epsilon_decay_multiplier: float = 0.999
    selection_index_divisor: int = 50


def validate_actor_learner_rl_config(config: ActorLearnerRLConfig) -> None:
    from bullet_hell_rl.net import protocol

    if config.action_dimension != protocol.FLAT_ACTION_COUNT:
        raise ValueError(
            f"action_dimension={config.action_dimension} must equal "
            f"net.protocol.FLAT_ACTION_COUNT={protocol.FLAT_ACTION_COUNT}"
        )
    if config.weights_publish_every < 1:
        raise ValueError(
            f"weights_publish_every={config.weights_publish_every} must be >= 1"
        )


ACTOR_LEARNER_RL_CONFIG = ActorLearnerRLConfig()
validate_actor_learner_rl_config(ACTOR_LEARNER_RL_CONFIG)

__all__ = [
    "ActorLearnerRLConfig",
    "ACTOR_LEARNER_RL_CONFIG",
    "validate_actor_learner_rl_config",
]
