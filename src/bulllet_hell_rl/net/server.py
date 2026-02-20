"""
Authoritative multiplayer Bullet Hell server.
Manages client connections, validates join, runs shared world simulation, broadcasts updates.

The server runs HEADLESS (no window, no rendering). Only clients render the game;
the server only simulates and sends state updates.
"""
import math
import os
import random
import socket
import threading
import time
from typing import Any

# Headless: no display for server
# os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from ..bullethell import (
    BULLET_DAMAGE,
    ENEMY_SPAWN_MAX,
    ENEMY_SPAWN_MIN,
    ENTITY_SIZE,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    Bullet,
    Enemy,
    Player,
)

from .protocol import (
    FLAT_ACTION_COUNT,
    MSG_ACTION,
    MSG_JOIN,
    MSG_REJECT,
    MSG_UPDATE,
    MSG_WELCOME,
    MOVE_LOOKUP,
    flat_action_to_move_and_angle,
    recv_message,
    send_message,
)


def _nearest_player(enemy: Enemy, players: dict[int, Player]) -> Player | None:
    """Return the living player nearest to the enemy (by center distance)."""
    if not players:
        return None
    ex = enemy.x + enemy.size // 2
    ey = enemy.y + enemy.size // 2
    best = None
    best_d2 = float("inf")
    for p in players.values():
        if p.health <= 0:
            continue
        px = p.x + p.size // 2
        py = p.y + p.size // 2
        d2 = (px - ex) ** 2 + (py - ey) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = p
    return best


class ClientRecord:
    def __init__(self, client_id: int, sock: socket.socket, player: Player):
        self.client_id = client_id
        self.sock = sock
        self.player = player
        self.latest_action: int = 0  # flat 0-19; default no move + 0°
        self.disconnected = False


def _client_recv_loop(
    client_id: int,
    sock: socket.socket,
    message_queue: list[tuple[int, dict | None]],
    queue_lock: threading.Lock,
) -> None:
    """Run in thread: read messages and push (client_id, msg) to queue; None on disconnect."""
    try:
        while True:
            msg = recv_message(sock)
            with queue_lock:
                message_queue.append((client_id, msg))
            if msg is None:
                break
    except OSError:
        with queue_lock:
            message_queue.append((client_id, None))
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _build_player_state(player: Player, client_id: int) -> dict[str, Any]:
    return {
        "id": client_id,
        "x": player.x,
        "y": player.y,
        "health": player.health,
        "size": player.size,
    }


def _build_bullet_state(bullet: Bullet) -> dict[str, Any]:
    return {
        "x": bullet.x,
        "y": bullet.y,
        "vel_x": bullet.vel_x,
        "vel_y": bullet.vel_y,
        "is_friendly": bullet.is_friendly,
        "size": bullet.size,
    }


def _build_enemy_state(enemy: Enemy) -> dict[str, Any]:
    return {"x": enemy.x, "y": enemy.y, "size": enemy.size}


def run_server(host: str = "0.0.0.0", port: int = 5555, secret: str | None = None) -> None:
    """
    Run the game server. Listens on host:port.
    If secret is set, clients must send {"type": "join", "token": secret}.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen()
    print(f"Server listening on {host}:{port}")

    clients: dict[int, ClientRecord] = {}
    next_client_id = 0
    clients_lock = threading.Lock()
    message_queue: list[tuple[int, dict | None]] = []
    queue_lock = threading.Lock()

    # Shared world (only modified on main thread during tick)
    enemies: list[Enemy] = []
    bullets: list[Bullet] = []
    last_enemy_spawn_time = 0
    next_spawn_interval = random.randint(ENEMY_SPAWN_MIN, ENEMY_SPAWN_MAX)
    current_time = 0
    TICK_RATE = 60
    delta_time = 1.0 / TICK_RATE

    # Spawn initial enemies (same as env reset: 3 enemies)
    for _ in range(3):
        ex = random.randint(0, WORLD_WIDTH - ENTITY_SIZE)
        ey = random.randint(0, WORLD_HEIGHT - ENTITY_SIZE)
        enemies.append(Enemy(ex, ey))

    def accept_loop() -> None:
        nonlocal next_client_id
        while True:
            try:
                sock, _ = listener.accept()
            except OSError:
                break
            with clients_lock:
                cid = next_client_id
                next_client_id += 1
            # Validate: expect one "join" message
            try:
                sock.settimeout(10.0)
                msg = recv_message(sock)
                sock.settimeout(None)
            except OSError:
                sock.close()
                continue
            if msg is None or msg.get("type") != MSG_JOIN:
                send_message(sock, {"type": MSG_REJECT, "reason": "expected join"})
                sock.close()
                continue
            if secret is not None and msg.get("token") != secret:
                send_message(sock, {"type": MSG_REJECT, "reason": "invalid token"})
                sock.close()
                continue
            # Spawn player at random position (same bounds as BulletHellEnv.reset)
            px = random.randint(0, WORLD_WIDTH - ENTITY_SIZE)
            py = random.randint(0, WORLD_HEIGHT - ENTITY_SIZE)
            player = Player(px, py, is_env=True)
            with clients_lock:
                clients[cid] = ClientRecord(cid, sock, player)
            send_message(sock, {
                "type": MSG_WELCOME,
                "client_id": cid,
                "x": float(player.x),
                "y": float(player.y),
                "world_width": WORLD_WIDTH,
                "world_height": WORLD_HEIGHT,
                "entity_size": ENTITY_SIZE,
            })
            t = threading.Thread(
                target=_client_recv_loop,
                args=(cid, sock, message_queue, queue_lock),
                daemon=True,
            )
            t.start()

    accept_thread = threading.Thread(target=accept_loop, daemon=True)
    accept_thread.start()

    try:
        while True:
            # Drain message queue and update latest_action or mark disconnect
            with queue_lock:
                to_process = message_queue[:]
                message_queue.clear()
            for cid, msg in to_process:
                if msg is None:
                    with clients_lock:
                        if cid in clients:
                            clients[cid].disconnected = True
                    continue
                if msg.get("type") == MSG_ACTION:
                    a = msg.get("action", 0)
                    if isinstance(a, int) and 0 <= a < FLAT_ACTION_COUNT:
                        with clients_lock:
                            if cid in clients:
                                clients[cid].latest_action = a

            # Remove disconnected clients
            with clients_lock:
                to_remove = [cid for cid, rec in clients.items() if rec.disconnected]
                for cid in to_remove:
                    del clients[cid]

            current_time += delta_time * 1000  # milliseconds for shoot timers

            # Enemy spawn
            if current_time - last_enemy_spawn_time >= next_spawn_interval:
                ex = random.randint(0, WORLD_WIDTH - ENTITY_SIZE)
                ey = random.randint(0, WORLD_HEIGHT - ENTITY_SIZE)
                enemies.append(Enemy(ex, ey))
                last_enemy_spawn_time = current_time
                next_spawn_interval = random.randint(ENEMY_SPAWN_MIN, ENEMY_SPAWN_MAX)

            with clients_lock:
                player_list = list(clients.values())
            if not player_list:
                time.sleep(delta_time)
                continue

            # Apply player actions and shoot
            for rec in player_list:
                move_idx, angle = flat_action_to_move_and_angle(rec.latest_action)
                dir_str = MOVE_LOOKUP[move_idx]
                rec.player.aim_angle = angle
                bullets_to_remove = rec.player.update(
                    delta_time, None, bullets, action=dir_str
                )
                for b in bullets_to_remove:
                    if b in bullets:
                        bullets.remove(b)
                pb = rec.player.shoot(current_time)
                if pb:
                    bullets.append(pb)

            # Enemies: target nearest player
            for enemy in enemies[:]:
                target = _nearest_player(enemy, {r.client_id: r.player for r in player_list})
                if target is None:
                    enemy.update_position(delta_time)
                    continue
                enemy_bullet = enemy.shoot(current_time)
                if enemy_bullet:
                    bullets.append(enemy_bullet)
                bullets_to_remove = enemy.update(
                    delta_time, target, current_time, bullets
                )
                for b in bullets_to_remove:
                    if b in bullets:
                        bullets.remove(b)
                if enemy.health <= 0:
                    enemies.remove(enemy)

            # Bullets
            for bullet in bullets[:]:
                bullet.update(delta_time)
                if bullet.is_off_screen():
                    bullets.remove(bullet)

            # Build and send update per client
            with clients_lock:
                still_connected = list(clients.values())
            for rec in still_connected:
                you = rec.player
                other_players = [
                    _build_player_state(r.player, r.client_id)
                    for r in still_connected
                    if r.client_id != rec.client_id
                ]
                payload = {
                    "type": MSG_UPDATE,
                    "you": _build_player_state(you, rec.client_id),
                    "players": other_players,
                    "enemies": [_build_enemy_state(e) for e in enemies],
                    "bullets": [_build_bullet_state(b) for b in bullets],
                }
                try:
                    send_message(rec.sock, payload)
                except OSError:
                    rec.disconnected = True

            time.sleep(delta_time)
    except KeyboardInterrupt:
        print("Server shutting down.")
    finally:
        listener.close()
        with clients_lock:
            for rec in list(clients.values()):
                try:
                    rec.sock.close()
                except OSError:
                    pass
