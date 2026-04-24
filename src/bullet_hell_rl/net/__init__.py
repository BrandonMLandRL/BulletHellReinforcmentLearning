"""Multiplayer client-server networking for Bullet Hell."""

from .client import run_client
from .protocol import (
    FLAT_ACTION_COUNT,
    flat_action_to_move_and_angle,
    move_and_angle_to_flat_action,
)
from .server import run_server

__all__ = [
    "run_client",
    "run_server",
    "run_actor",
    "FLAT_ACTION_COUNT",
    "flat_action_to_move_and_angle",
    "move_and_angle_to_flat_action",
]


def __getattr__(name: str):
    if name == "run_actor":
        from .actor import run_actor

        return run_actor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
