"""
Multiplayer Bullet Hell client.
Connects to server, receives welcome and state updates, sends actions, renders from server state.
"""
import os
import socket
import threading
from typing import Any
import queue
import pygame

from .protocol import (
    MSG_ACTION,
    MSG_EXPERIENCE_TUPLE,
    MSG_JOIN,
    MSG_RESPAWN,
    MSG_UPDATE,
    MSG_WELCOME,
    MSG_WEIGHTS_READY,
    recv_message,
    send_message,
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

from ..DQN.actor_learner_rl_config import ACTOR_LEARNER_RL_CONFIG, ActorLearnerRLConfig
from ..DQN.ActorServerComponent import Actor
from ..DQN.actor_learner_rl_bridge import (
    build_obs_from_update,
    compute_reward_and_done,
)
from ..DQN.actor_learner_metrics import (
    actor_metrics_log_interval,
    log_metrics_record,
    log_rl_config_snapshot,
)


def initialize_prediction_network(
    weights_path: str,
    bootstrap_weights_path: str | None,
    rl_config: ActorLearnerRLConfig | None = None,
) -> Actor:
    return Actor(
        on_message_callback=None,
        weights_path=weights_path,
        bootstrap_weights_path=bootstrap_weights_path,
        rl_config=rl_config or ACTOR_LEARNER_RL_CONFIG,
    )

# Use poll_message for a nonblocking queue check (e.g. MSG_WEIGHTS_READY from learner broadcasts).
def poll_message(actor, timeout=None):
    """Grab next message from learner recv queue, or None if empty / timeout."""
    try:
        return actor._recv_queue.get(timeout=timeout)
    except queue.Empty:
        return None
        
def send_experience(actor, state, action, reward, next_state, done, meta=None):
    """
    Queue an experience tuple to be delivered to the learner.
    Shape/encoding is up to you – just be consistent on learner side.
    """
    if actor._send_thread is None:
        return
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


def _drain_learner_messages(
    dqn_actor: Actor,
    weights_path: str,
) -> None:
    while True:
        try:
            msg = dqn_actor._recv_queue.get_nowait()
        except queue.Empty:
            break
        if msg.get("type") == MSG_WEIGHTS_READY:
            path = msg.get("path") or weights_path
            abs_path = path if os.path.isabs(path) else os.path.abspath(path)
            dqn_actor.reload_weights(abs_path)
            dqn_actor.send_weights_ack()


def run_actor(
    host: str = "127.0.0.1",
    port: int = 5555,
    token: str | None = None,
    weights_path: str = "shared_weights.h5",
    bootstrap_weights_path: str | None = None,
    rl_config: ActorLearnerRLConfig | None = None,
) -> None:
    """
    Connect to the game server, then run the render loop and send actions.
    A Pygame window is shown immediately so you always see something; it shows
    "Connecting..." then the game once the server sends updates.
    """

    cfg = rl_config or ACTOR_LEARNER_RL_CONFIG
    dqn_actor = initialize_prediction_network(
        weights_path, bootstrap_weights_path, rl_config=cfg
    )
    try:
        log_rl_config_snapshot("actor", cfg)
    except Exception as e:
        print(f"Actor: metrics config log failed: {e}")

    if dqn_actor.learner_socket is not None:
        dqn_actor.send_actor_ready()
    else:
        print("Actor: no learner connection (is the learner running on 127.0.0.1:5556?)")

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
    prev_update: dict[str, Any] | None = None
    prev_obs = None
    prev_action = DEFAULT_FLAT_ACTION
    rl_step = 0
    actor_r_sum = 0.0
    actor_r_n = 0
    life_episode_reward_sum = 0.0
    life_episode_rl_steps = 0
    actor_metric_every = actor_metrics_log_interval()

    while running:
        _drain_learner_messages(dqn_actor, weights_path)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        if not running:
            break

        now = pygame.time.get_ticks() / 1000.0

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

        curr_obs = build_obs_from_update(state)
        if prev_obs is not None and prev_update is not None:
            prev_tick = prev_update.get("tick")
            curr_tick = state.get("tick")
            tick_delta = None
            if isinstance(prev_tick, int) and isinstance(curr_tick, int):
                tick_delta = curr_tick - prev_tick
            reward, done, meta = compute_reward_and_done(prev_update, state, tick_delta)
            you_prev = prev_update.get("you") or {}
            you_curr = state.get("you") or {}
            h_prev = float(you_prev.get("health", 0.0))
            h_curr = float(you_curr.get("health", 0.0))

            send_experience(
                dqn_actor,
                prev_obs,
                prev_action,
                reward,
                curr_obs,
                done,
                meta,
            )
            rl_step += 1
            actor_r_sum += float(reward)
            actor_r_n += 1

            if h_prev > 0:
                life_episode_reward_sum += float(reward)
                life_episode_rl_steps += 1

            if h_curr <= 0.0 and h_prev > 0.0:
                try:
                    log_metrics_record(
                        "actor",
                        "life_end",
                        {
                            "life_episode_reward": life_episode_reward_sum,
                            "life_episode_rl_steps": life_episode_rl_steps,
                        },
                    )
                except Exception as e:
                    print(f"Actor: metrics log failed: {e}")
                life_episode_reward_sum = 0.0
                life_episode_rl_steps = 0

        selection_index = max(1, rl_step // cfg.selection_index_divisor)
        flat = int(dqn_actor.selectAction(curr_obs, selection_index))

        if rl_step > 0 and rl_step % actor_metric_every == 0:
            branches = dqn_actor.consume_branch_counts_window()
            total_b = sum(branches.values()) or 1
            mean_rw = (actor_r_sum / actor_r_n) if actor_r_n else 0.0
            try:
                log_metrics_record(
                    "actor",
                    "step_sample",
                    {
                        "rl_step": rl_step,
                        "selection_index": selection_index,
                        "epsilon": float(dqn_actor.epsilon),
                        "learner_connected": dqn_actor.learner_socket is not None,
                        "branch_counts": branches,
                        "frac_greedy": branches.get("greedy", 0) / total_b,
                        "frac_random": branches.get("random", 0) / total_b,
                        "frac_warmup": branches.get("warmup", 0) / total_b,
                        "mean_reward_window": mean_rw,
                        "reward_count_window": actor_r_n,
                        "last_action_branch": dqn_actor._last_action_branch,
                    },
                )
            except Exception as e:
                print(f"Actor: metrics log failed: {e}")
            actor_r_sum = 0.0
            actor_r_n = 0

        if now - last_send_time >= send_interval:
            try:
                send_message(sock, {"type": MSG_ACTION, "action": flat})
                last_send_time = now
            except OSError:
                break

        prev_update = dict(state)
        prev_obs = curr_obs
        prev_action = flat

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

