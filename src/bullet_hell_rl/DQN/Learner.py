# Learner TCP service: receives experience tuples from actors, trains legacy DQN, publishes weights.
import os
import queue
import threading

import numpy as np
from tensorflow import keras

from bullet_hell_rl.DQN.LearnerServerComponent import (
    HOST,
    PORT,
    LearnerServerComponent,
)
from bullet_hell_rl.DQN.DQNLegacy import DeepQLearning
from bullet_hell_rl.DQN.actor_learner_rl_bridge import validate_experience_shape
from bullet_hell_rl.net.protocol import (
    MSG_EXPERIENCE_TUPLE,
    MSG_WEIGHTS_READY,
    MSG_WEIGHTS_READY_ACK,
)


class _DummyEnv:
    """DeepQLearning only stores env for unused single-process training paths."""

    pass


def _atomic_save_model(model, path: str) -> None:
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    model.save(tmp)
    os.replace(tmp, path)


class Learner:
    def __init__(
        self,
        lsc: LearnerServerComponent,
        weights_path: str,
        gamma: float = 0.99,
        bootstrap_weights_path: str | None = None,
    ):
        self.lsc = lsc
        self.weights_path = os.path.abspath(weights_path)
        self.bootstrap_weights_path = (
            os.path.abspath(bootstrap_weights_path) if bootstrap_weights_path else None
        )
        self.stop_event = threading.Event()
        self._ack_lock = threading.Lock()
        self._can_broadcast = True
        self._pending_acks = 0

        self.dqn = DeepQLearning(
            _DummyEnv(),
            gamma=gamma,
            epsilon=1.0,
            numberEpisodes=10**9,
            modelFileName=self.weights_path,
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
            loaded = keras.models.load_model(load_path, compile=False)
            self.dqn.mainNetwork.set_weights(loaded.get_weights())
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
                return
        with self.lsc._client_lock:
            n = len(self.lsc._client_sockets)
        _atomic_save_model(self.dqn.mainNetwork, self.weights_path)
        if n == 0:
            print(f"Learner: saved weights to {self.weights_path} (no actors connected)")
            return
        with self._ack_lock:
            if not self._can_broadcast:
                return
            with self.lsc._client_lock:
                n = len(self.lsc._client_sockets)
            if n == 0:
                return
            self._can_broadcast = False
            self._pending_acks = n
        self.lsc._send_queue.put(
            {"type": MSG_WEIGHTS_READY, "path": self.weights_path}
        )
        print(f"Learner: broadcast weights_ready to {n} actor(s)")

    def run_training_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                msg = self.lsc._recv_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not isinstance(msg, dict):
                continue
            mtype = msg.get("type")
            if mtype == MSG_EXPERIENCE_TUPLE:
                try:
                    validate_experience_shape(msg)
                except ValueError as e:
                    print(f"Learner: bad experience: {e}")
                    continue
                state = msg["state"]
                action = int(msg["action"])
                reward = float(msg["reward"])
                next_state = msg["next_state"]
                done = bool(msg["done"])
                s = np.asarray(state, dtype=np.float32)
                ns = np.asarray(next_state, dtype=np.float32)
                self.dqn.replayBuffer.append((s, action, reward, ns, done))
                self.dqn.stepCount += 1
                trained = self.dqn.trainNetwork()
                if trained:
                    self._try_publish_weights()
            elif mtype == MSG_WEIGHTS_READY_ACK:
                self._on_ack()
            else:
                print(f"Unexpected Learner msg type: {mtype}")


def main(
    host: str = HOST,
    port: int = PORT,
    weights_path: str | None = None,
    bootstrap_weights_path: str | None = None,
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
