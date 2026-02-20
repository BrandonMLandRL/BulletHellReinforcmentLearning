#!/usr/bin/env python3
"""Run the Bullet Hell multiplayer client."""
import argparse
import sys
from pathlib import Path

# Ensure src is on path when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bulllet_hell_rl.net import run_client

import os
# at top of run_client(), before pygame.init():
if os.name == "nt":
    os.environ["SDL_VIDEODRIVER"] = "windows"
def main() -> None:
    p = argparse.ArgumentParser(description="Bullet Hell multiplayer client")
    p.add_argument("--host", default="127.0.0.1", help="Server host")
    p.add_argument("--port", type=int, default=5555, help="Server port")
    p.add_argument("--token", default=None, help="Optional join token if server requires it")
    args = p.parse_args()
    run_client(host=args.host, port=args.port, token=args.token)


if __name__ == "__main__":
    main()
