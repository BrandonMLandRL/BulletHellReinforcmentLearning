import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
import random
import socket
import threading
from typing import Any, Optional
import multiprocessing
import pygame

from bullet_hell_rl.net.protocol import (
    MSG_ACTION,
    MSG_JOIN,
    MSG_RESPAWN,
    MSG_UPDATE,
    MSG_WELCOME,
    move_and_angle_to_flat_action,
    recv_message,
    send_message,
)
# from typing import Optional, Union

from bullet_hell_rl.bullethell import * 


"""

This Env will follow the Actor - Learner Archictecture. The Env instance will serve as the Actor. The Neural 
Network Implementation will be the Learner.

The Multiplayer Bullet Hell Env has these main differences from the standalone singleplayer edition.

1) It relies on a connection to the game server to receive data. The game server MUST be running.

2) It expects two pieces of memory intended to be shared via IPC: A queue that it will fill to send observations
 to the Network (Learner) for training and reference to the weights that the Learner updates.

3) This Actor will only make predictions. It will enqueue (currentState,action,reward,nextState,terminalState). 
Thus is will save previous state, which is also unlike singleplayer.

4) When render mode is human, it will boot up a rendering Thread, that renders the last received UPDATE packet. 

"""

server_details = {
    "host": "127.0.0.1",
    "port": 5555,
}
metadata = {
    "render_modes": ["human", "terminal"],
    "render_fps": 60,
}
class MultiplayerBulletHellEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self, render_mode: str | None = "terminal", shared_queue = multiprocessing.Queue(), init_network = None, weights_reference = None):
        #Define Action Space
        #Define Observation Space


        self.game_server_handshake()
        assert all(
            isinstance(x, int) for x in (
                self.client_id,
                self.world_width,
                self.world_height,
                self.entity_size,
                self.bullet_size,
            )
        ), "welcome packet values (client_id, world_width, world_height, entity_size, bullet_size) must be integers"

        self.last_state: dict[str, Any] = {}
        
        self.render_mode = render_mode
        if self.render_mode == "human":
            pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):        #
        pass
    
    def step(self, action): 
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        #TODO: Implement recv_loop
        #If incoming message is Update: Build the enqueue package for the Learner

        #If the incoming message is Respawn, build the enqueue package for the learner and terminate the episode.
        

    def render(self):
       #Todo: Implement Threading
       pass

    def close(self):
        if self.screen is not None:

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def game_server_handshake(self):
         #Connect to Game Server
        try:
            print("atempting to connect to server")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((server_details["host"], server_details["port"]))
            sock.settimeout(None)
    
        except (OSError, socket.error) as e:
            if not print(f"Connection failed: {e}"):
                pygame.quit()
                raise
            pygame.quit()
            raise RuntimeError(f"Cannot connect to {server_details['host']}:{server_details['port']}") from e

        send_message(sock, {"type": MSG_JOIN, **({})})
        welcome = recv_message(sock)
        if welcome is None or welcome.get("type") != MSG_WELCOME:
            sock.close()
            if not print("Server rejected connection or invalid welcome."):
                pygame.quit()
                raise
            pygame.quit()
            raise RuntimeError("Server rejected connection or sent invalid welcome")

        self.client_id = welcome["client_id"]
        self.world_width = welcome.get("world_width", 1000)
        self.world_height = welcome.get("world_height", 1000)
        self.entity_size = welcome.get("entity_size", 20)
        self.bullet_size = 10  # default if not in welcome

if __name__ == "__main__":
    env = MultiplayerBulletHellEnv()