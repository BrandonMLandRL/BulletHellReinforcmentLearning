#!/usr/bin/env python3
"""Run the Bullet Hell learner server."""
import argparse
import sys
from pathlib import Path

# Ensure src is on path when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bullet_hell_rl.DQN.Learner import main as learner_main


def main() -> None:
    p = argparse.ArgumentParser(description="Bullet Hell learner server")
    p.add_argument("--host", default="127.0.0.1", help="Bind host")
    p.add_argument("--port", type=int, default=5556, help="Bind port")
    args = p.parse_args()
    learner_main(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
