# Launch game server, learner, and actor subprocesses with a shared weights file path.
import os
import subprocess
import sys
import time


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    weights_name = os.environ.get("SHARED_WEIGHTS", "shared_weights.h5")
    weights_path = weights_name if os.path.isabs(weights_name) else os.path.join(root, weights_name)

    bootstrap = os.environ.get("BOOTSTRAP_WEIGHTS")
    if bootstrap and not os.path.isabs(bootstrap):
        bootstrap = os.path.join(root, bootstrap)

    env = os.environ.copy()
    env["SHARED_WEIGHTS"] = weights_path
    if bootstrap:
        env["BOOTSTRAP_WEIGHTS"] = bootstrap

    creationflags = subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
    popen_kw = {"env": env, "cwd": root}
    if creationflags:
        popen_kw["creationflags"] = creationflags

    subprocess.Popen([sys.executable, "-u", "run_server.py"], **popen_kw)
    # Learner must listen before actors connect (short timeout in Actor.__init__).
    subprocess.Popen([sys.executable, "-u", "run_learner.py", "--weights", weights_path], **popen_kw)
    time.sleep(2.5)
    actor_cmd = [sys.executable, "-u", "run_actor.py", "--weights", weights_path]
    for i in range(3):

        if bootstrap:
            actor_cmd.extend(["--bootstrap", bootstrap])
        subprocess.Popen(actor_cmd, **popen_kw)


if __name__ == "__main__":
    main()
