# Learner TCP service: receives experience tuples from actors, trains legacy DQN, publishes weights.
import gc
import os
import queue
import shutil
import threading
import time

import numpy as np
from tensorflow import keras

from bullet_hell_rl.DQN.LearnerServerComponent import (
    HOST,
    PORT,
    LearnerServerComponent,
)
from bullet_hell_rl.DQN.actor_learner_rl_config import (
    ACTOR_LEARNER_RL_CONFIG,
    ActorLearnerRLConfig,
)
from bullet_hell_rl.DQN.DQNLegacy import DeepQLearning
from bullet_hell_rl.DQN.actor_learner_rl_bridge import validate_experience_shape
from bullet_hell_rl.net.protocol import (
    MSG_ACTOR_READY,
    MSG_EXPERIENCE_TUPLE,
    MSG_WEIGHTS_READY,
    MSG_WEIGHTS_READY_ACK,
)


def _log_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


class _DummyEnv:
    """DeepQLearning only stores env for unused single-process training paths."""

    pass


def _atomic_save_weights(model, path: str) -> None:
    """
    Write weights-only HDF5 to a unique temp file, then replace the destination.
    Temp paths must end in .h5 so Keras writes a single file (not a SavedModel folder).
    On Windows, replace fails if another process still has the file open; retry briefly.
    """
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{time.time_ns()}.h5"
    try:
        model.save_weights(tmp, save_format="h5")
        last_err: OSError | None = None
        for _ in range(60):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=False)
                os.replace(tmp, path)
                return
            except (PermissionError, OSError) as e:
                last_err = e
                time.sleep(0.05)
        raise last_err if last_err else OSError("failed to publish weights file")
    finally:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


class Learner:
    def __init__(
        self,
        lsc: LearnerServerComponent,
        weights_path: str,
        gamma: float | None = None,
        bootstrap_weights_path: str | None = None,
        rl_config: ActorLearnerRLConfig | None = None,
    ):
        self.lsc = lsc
        self.weights_path = os.path.abspath(weights_path)
        self.bootstrap_weights_path = (
            os.path.abspath(bootstrap_weights_path) if bootstrap_weights_path else None
        )
        cfg = rl_config or ACTOR_LEARNER_RL_CONFIG
        if gamma is None:
            gamma = cfg.gamma
        self.stop_event = threading.Event()
        self._ack_lock = threading.Lock()
        self._can_broadcast = True
        self._pending_acks = 0
        self._experience_recv_count = 0
        self._weights_publish_every = cfg.weights_publish_every
        self._weights_publish_train_counter = 0

        self.dqn = DeepQLearning(
            _DummyEnv(),
            gamma=gamma,
            epsilon=cfg.epsilon_start,
            numberEpisodes=cfg.number_episodes,
            modelFileName=self.weights_path,
            rl_config=cfg,
        )
        self._bootstrap_or_load_weights()

    def _bootstrap_or_load_weights(self) -> None:
        load_path = None
        if os.path.isfile(self.weights_path):
            load_path = self.weights_path
        elif self.bootstrap_weights_path and os.path.isfile(self.bootstrap_weights_path):
            load_path = self.bootstrap_weights_path
        if load_path is None:
            print(f"Learner: no checkpoint at {self.weights_path}; using fresh networks")
            return
        try:
            try:
                self.dqn.mainNetwork.load_weights(load_path)
            except Exception:
                loaded = keras.models.load_model(load_path, compile=False)
                self.dqn.mainNetwork.set_weights(loaded.get_weights())
                del loaded
                gc.collect()
            self.dqn.targetNetwork.set_weights(self.dqn.mainNetwork.get_weights())
            print(f"Learner: loaded weights from {load_path}")
        except Exception as e:
            print(f"Learner: could not load {load_path}: {e}")

    def _on_ack(self) -> None:
        with self._ack_lock:
            if self._pending_acks > 0:
                self._pending_acks -= 1
            if self._pending_acks <= 0:
                self._pending_acks = 0
                self._can_broadcast = True

    def _try_publish_weights(self) -> None:
        with self._ack_lock:
            if not self._can_broadcast:
                # print(
                #     f"[{_log_ts()}] Learner [weights-publish]: skip — waiting for "
                #     f"{self._pending_acks} actor ACK(s) from last weights_ready "
                #     f"(cannot start another broadcast until all ACK)",
                #     flush=True,
                # )
                return
        with self.lsc._client_lock:
            n_clients_before = len(self.lsc._client_sockets)
        t_save0 = time.perf_counter()
        try:
            _atomic_save_weights(self.dqn.mainNetwork, self.weights_path)
        except (PermissionError, OSError) as e:
            print(
                f"Learner: failed to save weights to {self.weights_path}: {e!r}. "
                "Close other programs using this file; Actor releases the handle after each load_weights.",
                flush=True,
            )
            return
        save_s = time.perf_counter() - t_save0
        if n_clients_before == 0:
            print(
                f"[{_log_ts()}] Learner [weights-publish]: saved {self.weights_path} "
                f"(no actors connected, save {save_s:.3f}s) — no broadcast",
                flush=True,
            )
            return
        with self._ack_lock:
            if not self._can_broadcast:
                print(
                    f"[{_log_ts()}] Learner [weights-publish]: saved ({save_s:.3f}s) but skip broadcast — "
                    f"can_broadcast became False (pending_acks={self._pending_acks})",
                    flush=True,
                )
                return
            with self.lsc._client_lock:
                n = len(self.lsc._client_sockets)
            if n == 0:
                print(
                    f"[{_log_ts()}] Learner [weights-publish]: saved ({save_s:.3f}s) but skip broadcast — "
                    f"actors went from {n_clients_before} to 0 during save",
                    flush=True,
                )
                return
            self._can_broadcast = False
            self._pending_acks = n
        self.lsc._send_queue.put(
            {"type": MSG_WEIGHTS_READY, "path": self.weights_path}
        )
        print(
            f"[{_log_ts()}] Learner [weights-publish]: broadcast weights_ready to {n} actor(s) "
            f"(save {save_s:.3f}s, awaiting {n} ACK)",
            flush=True,
        )

    _EXPERIENCE_GET_TIMEOUT_SEC = 0.05

    def _drain_priority_inbound(self) -> None:
        """Process all pending ACK / handshake messages before experience work."""
        while True:
            try:
                msg = self.lsc._recv_priority_queue.get_nowait()
            except queue.Empty:
                break
            if not isinstance(msg, dict):
                continue
            self._dispatch_priority_message(msg)
        self.lsc._recv_priority_pending.clear()

    def _dispatch_priority_message(self, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == MSG_ACTOR_READY:
            print(
                f"[{_log_ts()}] Learner: actor_ready (handshake)",
                flush=True,
            )
        elif mtype == MSG_WEIGHTS_READY_ACK:
            self._on_ack()
            with self._ack_lock:
                pend = self._pending_acks
                can_b = self._can_broadcast
            print(
                f"[{_log_ts()}] Learner: received weights_ack from actor "
                f"(pending_acks={pend}, can_broadcast={can_b})",
                flush=True,
            )
        else:
            print(f"Unexpected Learner priority msg type: {mtype}")

    def _handle_experience_message(self, msg: dict) -> None:
        try:
            validate_experience_shape(msg)
        except ValueError as e:
            print(f"Learner: bad experience: {e}")
            return
        state = msg["state"]
        action = int(msg["action"])
        reward = float(msg["reward"])
        next_state = msg["next_state"]
        done = bool(msg["done"])
        s = np.asarray(state, dtype=np.float32)
        ns = np.asarray(next_state, dtype=np.float32)
        self.dqn.replayBuffer.append((s, action, reward, ns, done))
        self.dqn.stepCount += 1
        self._experience_recv_count += 1
        trained = self.dqn.trainNetwork()
        if trained:
            print(
                f"[{_log_ts()}] Learner: trained network "
                f"(step_count={self.dqn.stepCount}, replay_size={len(self.dqn.replayBuffer)})",
                flush=True,
            )
            self._weights_publish_train_counter += 1
            if self._weights_publish_train_counter < self._weights_publish_every:
                return
            self._weights_publish_train_counter = 0
            self._try_publish_weights()

    def run_training_loop(self) -> None:
        while not self.stop_event.is_set():
            self._drain_priority_inbound()
            try:
                msg = self.lsc._recv_queue.get(timeout=self._EXPERIENCE_GET_TIMEOUT_SEC)
            except queue.Empty:
                continue
            if not isinstance(msg, dict):
                continue
            mtype = msg.get("type")
            if mtype == MSG_EXPERIENCE_TUPLE:
                self._handle_experience_message(msg)
            else:
                print(f"Unexpected Learner msg type on experience queue: {mtype}")


def main(
    host: str = HOST,
    port: int = PORT,
    weights_path: str | None = None,
    bootstrap_weights_path: str | None = None,
    rl_config: ActorLearnerRLConfig | None = None,
) -> None:
    weights_path = weights_path or os.environ.get("SHARED_WEIGHTS", "shared_weights.h5")
    bootstrap = bootstrap_weights_path or os.environ.get("BOOTSTRAP_WEIGHTS")
    if bootstrap and not os.path.isfile(bootstrap):
        bootstrap = None

    lsc = LearnerServerComponent(host=host, port=port)
    lsc.start_background()
    learner = Learner(
        lsc,
        weights_path=weights_path,
        bootstrap_weights_path=bootstrap,
        rl_config=rl_config,
    )
    try:
        learner.run_training_loop()
    except KeyboardInterrupt:
        print("Learner shutdown (KeyboardInterrupt)")
    finally:
        learner.stop_event.set()
        lsc.close()


if __name__ == "__main__":
    main()
