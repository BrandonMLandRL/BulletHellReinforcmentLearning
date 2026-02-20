"""
Multiplayer Bullet Hell client.
Connects to server, receives welcome and state updates, sends actions, renders from server state.
"""
import socket
import threading
from typing import Any

import pygame

from .protocol import (
    MSG_ACTION,
    MSG_JOIN,
    MSG_UPDATE,
    MSG_WELCOME,
    move_and_angle_to_flat_action,
    recv_message,
    send_message,
)

# Colors (match bullethell)
BLUE = (0, 100, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500

# Default: no move (4), aim 0° -> flat 16
DEFAULT_FLAT_ACTION = 4 * 4 + 0  # 16


def _keys_to_flat_action(keys: pygame.key.ScancodeWrapper) -> int:
    """Map keyboard state to flat action [0, 19]. Move from arrows/WASD, aim from 1-4 keys."""
    # Move: 0=left, 1=right, 2=up, 3=down, 4=none
    move = 4
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        move = 0
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        move = 1
    elif keys[pygame.K_UP] or keys[pygame.K_w]:
        move = 2
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
        move = 3
    # Aim: 0°, 90°, 180°, 270° from keys 1-4 (numkey or main row)
    aim_idx = 0
    if keys[pygame.K_2]:
        aim_idx = 1  # 90
    elif keys[pygame.K_3]:
        aim_idx = 2  # 180
    elif keys[pygame.K_4]:
        aim_idx = 3  # 270
    return move_and_angle_to_flat_action(move, aim_idx * 90)


def run_client(
    host: str = "127.0.0.1",
    port: int = 5555,
    token: str | None = None,
) -> None:
    """
    Connect to the game server, then run the render loop and send actions.
    A Pygame window is shown immediately so you always see something; it shows
    "Connecting..." then the game once the server sends updates.
    """
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
    world_width = welcome.get("world_width", 500)
    world_height = welcome.get("world_height", 500)
    entity_size = welcome.get("entity_size", 20)
    bullet_size = 10  # default if not in welcome

    # Shared state from server (updated by recv thread)
    last_state: dict[str, Any] = {}
    state_lock = threading.Lock()

    def recv_loop() -> None:
        while True:
            msg = recv_message(sock)
            if msg is None:
                break
            if msg.get("type") == MSG_UPDATE:
                with state_lock:
                    last_state.clear()
                    last_state.update(msg)
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

        keys = pygame.key.get_pressed()
        flat = _keys_to_flat_action(keys)
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
        print(f"game happening, {you})")

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
            color = BLUE if b.get("is_friendly", True) else RED
            pygame.draw.circle(
                screen, color,
                (int(bx + sz / 2), int(by + sz / 2)),
                max(1, sz // 2),
            )
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    try:
        sock.close()
    except OSError:
        pass

