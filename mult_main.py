#This file will be used to create and run the GameServer, Actor, and Learner
#It will also take in a filename to be used as the shared weights file between Actor and Learner
import os
import sys
import queue
import subprocess

def main() -> None:
    #Filename Arg Parse
    filename = "shared_weights.h5"

    #Create Queue for IPC
    q = queue.Queue()

    #Create GameServer
    # from src.bullet_hell_rl.net.server import run_server
    run_game_server = subprocess.Popen(
        [sys.executable, "-u", "run_server.py"], creationflags=subprocess.CREATE_NEW_CONSOLE
    )

    #Create Learner - Learner will accept a queue of experience tuples (s, a, r, s') from Actors
    run_learner = subprocess.Popen(
        [sys.executable, "-u", "run_learner.py"], creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    # Create x Actors - actor will automatically connect to GameServer
    run_actor = subprocess.Popen(
        [sys.executable, "-u", "run_actor.py"], creationflags=subprocess.CREATE_NEW_CONSOLE
    )

if __name__ == "__main__":
    main()