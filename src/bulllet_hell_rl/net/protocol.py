"""
Network protocol for Bullet Hell multiplayer.
Length-prefixed JSON messages; action encoding (5 move x 4 aim = 20 flat actions).
"""
import json
import struct
from typing import Any

# Message types
MSG_WELCOME = "welcome"
MSG_UPDATE = "update"
MSG_ACTION = "action"
MSG_JOIN = "join"
MSG_REJECT = "reject"

# Action space: 5 moves (0-4) x 4 aim angles (0, 90, 180, 270) = 20 flat actions
MOVE_LOOKUP = {
    0: "left",
    1: "right",
    2: "up",
    3: "down",
    4: "none",
}
NUM_MOVES = 5
NUM_AIM_ANGLES = 4
FLAT_ACTION_COUNT = NUM_MOVES * NUM_AIM_ANGLES  # 20


def flat_action_to_move_and_angle(flat_action: int) -> tuple[int, int]:
    """
    Map flat action index in [0, 19] to (move_index, fire_angle_degrees).
    move: 0=left, 1=right, 2=up, 3=down, 4=none.
    angle: 0, 90, 180, 270.
    """
    flat_action = max(0, min(19, int(flat_action)))
    move = flat_action // NUM_AIM_ANGLES
    angle = (flat_action % NUM_AIM_ANGLES) * 90
    return move, angle


def move_and_angle_to_flat_action(move: int, angle_degrees: int) -> int:
    """Map (move_index, fire_angle) to flat action in [0, 19]."""
    move = max(0, min(4, int(move)))
    # Snap angle to 0, 90, 180, 270
    angle_idx = round(angle_degrees / 90) % 4
    return move * NUM_AIM_ANGLES + angle_idx


# Length prefix: 4 bytes big-endian unsigned int (max message ~4GB)
LENGTH_PREFIX_BYTES = 4
LENGTH_PREFIX_FMT = ">I"


def send_message(sock, obj: dict[str, Any]) -> None:
    """Send a JSON-serializable dict as length-prefixed message."""
    payload = json.dumps(obj).encode("utf-8")
    length = len(payload)
    sock.sendall(struct.pack(LENGTH_PREFIX_FMT, length))
    sock.sendall(payload)


def recv_message(sock) -> dict[str, Any] | None:
    """
    Read one length-prefixed JSON message from the socket.
    Returns None on EOF/closed connection.
    """
    try:
        length_buf = _recv_exact(sock, LENGTH_PREFIX_BYTES)
        if not length_buf:
            return None
        (length,) = struct.unpack(LENGTH_PREFIX_FMT, length_buf)
        if length == 0 or length > 10 * 1024 * 1024:  # reject > 10MB
            return None
        payload_buf = _recv_exact(sock, length)
        if not payload_buf or len(payload_buf) != length:
            return None
        return json.loads(payload_buf.decode("utf-8"))
    except (json.JSONDecodeError, struct.error, OSError):
        return None


def _recv_exact(sock, n: int) -> bytes | None:
    """Read exactly n bytes; return None on EOF."""
    buf = []
    while n > 0:
        chunk = sock.recv(n)
        if not chunk:
            return None
        buf.append(chunk)
        n -= len(chunk)
    return b"".join(buf)
