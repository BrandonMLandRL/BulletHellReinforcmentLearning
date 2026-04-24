#!/usr/bin/env python3
"""Run the Bullet Hell multiplayer client."""
import argparse
import os
import sys
from pathlib import Path

# Ensure src is on path when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bullet_hell_rl.net import run_actor

import os
# at top of run_client(), before pygame.init():
if os.name == "nt":
    os.environ["SDL_VIDEODRIVER"] = "windows"
def main() -> None:
    p = argparse.ArgumentParser(description="Bullet Hell multiplayer client")
    p.add_argument("--host", default="127.0.0.1", help="Server host")
    p.add_argument("--port", type=int, default=5555, help="Server port")
    p.add_argument("--token", default=None, help="Optional join token if server requires it")
    p.add_argument(
        "--weights",
        default=None,
        help="Shared policy weights path (default: SHARED_WEIGHTS env or shared_weights.h5)",
    )
    p.add_argument(
        "--bootstrap",
        default=None,
        help="Optional legacy .h5 to load if shared weights are missing",
    )
    args = p.parse_args()
    weights = args.weights or os.environ.get("SHARED_WEIGHTS", "shared_weights.h5")
    bootstrap = args.bootstrap or os.environ.get("BOOTSTRAP_WEIGHTS")
    run_actor(
        host=args.host,
        port=args.port,
        token=args.token,
        weights_path=weights,
        bootstrap_weights_path=bootstrap,
    )


if __name__ == "__main__":
    main()
