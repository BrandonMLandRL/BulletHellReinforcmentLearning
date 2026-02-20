#!/usr/bin/env python3
"""Run the Bullet Hell multiplayer server."""
import argparse
import sys
from pathlib import Path

# Ensure src is on path when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bulllet_hell_rl.net import run_server


def main() -> None:
    p = argparse.ArgumentParser(description="Bullet Hell multiplayer server")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=5555, help="Bind port")
    p.add_argument("--secret", default=None, help="Optional token clients must send to join")
    args = p.parse_args()
    run_server(host=args.host, port=args.port, secret=args.secret)


if __name__ == "__main__":
    main()
