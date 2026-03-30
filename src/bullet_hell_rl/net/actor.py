"""
Multiplayer Bullet Hell client.
Connects to server, receives welcome and state updates, sends actions, renders from server state.
"""
import random
import socket
import threading
from typing import Any
import queue
import pygame

from .protocol import (
    MSG_ACTION,
    MSG_JOIN,
    MSG_RESPAWN,
    MSG_UPDATE,
    MSG_WELCOME,
    move_and_angle_to_flat_action,
    recv_message,
    send_message,
    MSG_EXPERIENCE_TUPLE,
    MSG_WEIGHTS_READY,
    MSG_WEIGHTS_READY_ACK
)

# Colors (match bullethell)
BLUE = (0, 100, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 50)

from ..bullethell import (
    SCREEN_HEIGHT,
    SCREEN_WIDTH
)


# Default: no move (4), aim 0° -> flat 16
DEFAULT_FLAT_ACTION = 4 * 4 + 0  # 16

from ..DQN.ActorComponent import Actor


def initialize_prediction_network():
    return Actor(on_message_callback=None)

#Use poll_message(0) for a nonblocking queue check - we will be looking for the message MSG_WEIGHTS_READY
#Should run this function and then if the queue has message weights ready 
def poll_message(actor, timeout=None):
    """Grab next message from learner recv queue, or None if empty / timeout."""
    try:
        msg = actor._recv_queue.get(timeout=timeout)
        print(msg)
        return msg
    except queue.Empty:
        return None
        
def send_experience(actor, state, action, reward, next_state, done, meta=None):
    """
    Queue an experience tuple to be delivered to the learner.
    Shape/encoding is up to you – just be consistent on learner side.
    """
    msg = {
        "type": MSG_EXPERIENCE_TUPLE,
        "state": state.tolist() if hasattr(state, "tolist") else state,
        "action": int(action),
        "reward": float(reward),
        "next_state": next_state.tolist() if hasattr(next_state, "tolist") else next_state,
        "done": bool(done),
        "meta": meta or {},
    }
    actor._send_queue.put(msg)

def send_weights_ack(actor):
    msg = {
        "type": MSG_WEIGHTS_READY_ACK
    }
    actor._send_queue.put(msg)
def run_actor(
    host: str = "127.0.0.1",
    port: int = 5555,
    token: str | None = None,
) -> None:
    """
    Connect to the game server, then run the render loop and send actions.
    A Pygame window is shown immediately so you always see something; it shows
    "Connecting..." then the game once the server sends updates.
    """

    dqn_actor = initialize_prediction_network()

    # Learner handshake: after welcome, learner sends MSG_WEIGHTS_READY; we ACK once.
    msg = poll_message(dqn_actor, timeout=5.0)
    if msg is not None and msg.get("type") == MSG_WEIGHTS_READY:
        print(f"Actor: received MSG_WEIGHTS_READY {msg}")
        send_weights_ack(dqn_actor)
    elif msg is not None:
        print(f"Actor: unexpected first message from learner (expected MSG_WEIGHTS_READY): {msg}")
    else:
        print("Actor: timed out waiting for MSG_WEIGHTS_READY (is the learner running on 127.0.0.1:5556?)")

    print("establishing pygame")
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Bullet Hell Multiplayer - Connecting...")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    def show_message(msg: str, duration_sec: float | None = None) -> bool:
        """Draw msg centered; if duration_sec, wait that long. Return False to exit."""
        t0 = pygame.time.get_ticks() / 1000.0
        while duration_sec is None or (pygame.time.get_ticks() / 1000.0 - t0) < duration_sec:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return False
            screen.fill(BLACK)
            text = font.render(msg, True, WHITE)
            r = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(text, r)
            pygame.display.flip()
            clock.tick(60)
            if duration_sec is not None and (pygame.time.get_ticks() / 1000.0 - t0) >= duration_sec:
                break
        return True

    try:
        print("atempting to connect to server")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((host, port))
        sock.settimeout(None)
    except (OSError, socket.error) as e:
        if not show_message(f"Connection failed: {e}", duration_sec=5.0):
            pygame.quit()
            raise
        pygame.quit()
        raise RuntimeError(f"Cannot connect to {host}:{port}") from e

    print("connected to server")
    send_message(sock, {"type": MSG_JOIN, **({"token": token} if token else {})})
    welcome = recv_message(sock)
    if welcome is None or welcome.get("type") != MSG_WELCOME:
        sock.close()
        if not show_message("Server rejected connection or invalid welcome.", duration_sec=5.0):
            pygame.quit()
            raise
        pygame.quit()
        raise RuntimeError("Server rejected connection or sent invalid welcome")

    pygame.display.set_caption("Bullet Hell Multiplayer")
    client_id = welcome["client_id"]
    world_width = welcome.get("world_width", 1000)
    world_height = welcome.get("world_height", 1000)
    entity_size = welcome.get("entity_size", 20)
    bullet_size = 10  # default if not in welcome

    # Shared state from server (updated by recv thread)
    last_state: dict[str, Any] = {}
    last_respawn: dict[str, Any] | None = None
    state_lock = threading.Lock()
    respawn_show_until = 0.0  # time (seconds) when to stop showing "Respawned!"

    def recv_loop() -> None:
        nonlocal last_respawn
        while True:
            msg = recv_message(sock)
            if msg is None:
                break
            if msg.get("type") == MSG_UPDATE:
                with state_lock:
                    last_state.clear()
                    last_state.update(msg)
                    # print(msg.get("tick"))
            elif msg.get("type") == MSG_RESPAWN:
                with state_lock:
                    last_respawn = msg
        try:
            sock.close()
        except OSError:
            pass

    recv_thread = threading.Thread(target=recv_loop, daemon=True)
    recv_thread.start()
    send_interval = 1.0 / 60  # send at most 60 actions per second
    last_send_time = 0.0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        if not running:
            break

        flat = random.randint(0, 19)
        now = pygame.time.get_ticks() / 1000.0
        if now - last_send_time >= send_interval:
            try:
                send_message(sock, {"type": MSG_ACTION, "action": flat})
                last_send_time = now
            except OSError:
                break

        with state_lock:
            state = dict(last_state)
        if not state or state.get("type") != MSG_UPDATE:
            screen.fill(BLACK)
            status = font.render("Waiting for game state...", True, WHITE)
            status_r = status.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(status, status_r)
            pygame.display.flip()
            clock.tick(60)
            continue

        you = state.get("you", {})
        now_sec = pygame.time.get_ticks() / 1000.0
        with state_lock:
            if last_respawn is not None:
                respawn_show_until = now_sec + 1.0
                last_respawn = None

        players = state.get("players", [])
        enemies = state.get("enemies", [])
        bullets = state.get("bullets", [])
        my_x = you.get("x", 0)
        my_y = you.get("y", 0)
        my_size = you.get("size", entity_size)
        camera_x = my_x + my_size // 2 - SCREEN_WIDTH // 2
        camera_y = my_y + my_size // 2 - SCREEN_HEIGHT // 2
        camera_x = max(0, min(world_width - SCREEN_WIDTH, camera_x))
        camera_y = max(0, min(world_height - SCREEN_HEIGHT, camera_y))

        screen.fill(BLACK)
        # Draw me
        sx = my_x - camera_x
        sy = my_y - camera_y
        pygame.draw.rect(screen, BLUE, (sx, sy, my_size, my_size))
        # Other players
        for p in players:
            px = p.get("x", 0) - camera_x
            py = p.get("y", 0) - camera_y
            sz = p.get("size", entity_size)
            pygame.draw.rect(screen, GREEN, (px, py, sz, sz))
        # Enemies
        for e in enemies:
            ex = e.get("x", 0) - camera_x
            ey = e.get("y", 0) - camera_y
            sz = e.get("size", entity_size)
            pygame.draw.rect(screen, RED, (ex, ey, sz, sz))
        # Bullets
        for b in bullets:
            bx = b.get("x", 0) - camera_x
            by = b.get("y", 0) - camera_y
            sz = b.get("size", bullet_size)
            # print(f"{b.get('owner_id')}  ?=?  {client_id}")
            if b.get("owner_id") == client_id:
                color = BLUE
            elif b.get("is_friendly", False):
                color = GREEN
            else:
                color = RED
            pygame.draw.circle(
                screen, color,
                (int(bx + sz / 2), int(by + sz / 2)),
                max(1, sz // 2),
            )
        if now_sec < respawn_show_until:
            respawn_text = font.render("Respawned!", True, WHITE)
            respawn_r = respawn_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(respawn_text, respawn_r)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    try:
        sock.close()
    except OSError:
        pass

