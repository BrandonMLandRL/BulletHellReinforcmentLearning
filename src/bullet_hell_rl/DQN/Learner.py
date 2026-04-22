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
        self._experience_recv_count = 0

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
                return
        with self.lsc._client_lock:
            n = len(self.lsc._client_sockets)
        try:
            _atomic_save_weights(self.dqn.mainNetwork, self.weights_path)
        except (PermissionError, OSError) as e:
            print(
                f"Learner: failed to save weights to {self.weights_path}: {e!r}. "
                "Close other programs using this file; Actor releases the handle after each load_weights.",
                flush=True,
            )
            return
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
                self._experience_recv_count += 1
                nrx = self._experience_recv_count
                buf = len(self.dqn.replayBuffer)
                # if nrx <= 5 or nrx % 30 == 0:
                #     print(
                #         f"Learner: received experience #{nrx} from actor "
                #         f"(action={action}, reward={reward:.2f}, done={done}, replay_size={buf})",
                #         flush=True,
                #     )
                trained = self.dqn.trainNetwork()
                if trained:
                    self._try_publish_weights()
            elif mtype == MSG_WEIGHTS_READY_ACK:
                self._on_ack()
                with self._ack_lock:
                    pend = self._pending_acks
                    can_b = self._can_broadcast
                print(
                    f"Learner: received weights_ack from actor "
                    f"(pending_acks={pend}, can_broadcast={can_b})",
                    flush=True,
                )
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
