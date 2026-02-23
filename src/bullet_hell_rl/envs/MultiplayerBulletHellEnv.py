import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
import random
import socket
import threading
from typing import Any
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

metadata = {
    "render_modes": ["human", "terminal"],
    "render_fps": 60,
}

"""
The Multiplayer Bullet Hell Env has a few main differences from the standalone singleplayer edition.

1) It relies on a connection to the game server to receive data. The game server MUST be running.
2) It expects two pieces of memory intended to be shared via IPC.
"""
class MultiplayerBulletHellEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self, render_mode: str | None = "terminal", shared_queue = multiprocessing.Queue(), init_network = None, weights_reference = None):
        #Define Action Space
        #Define Observation Space

        self.render_mode = render_mode



        #If render mode == human, create render thread
        #Render thread will use the data of our previous state to render. 
    def step(self, action): 
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        if self.render_mode == "human":
            self.render()

    def render(self):
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Bullet Hell Environment")

        camera_x = self.player.x + self.player.size // 2 - SCREEN_WIDTH // 2
        camera_y = self.player.y + self.player.size // 2 - SCREEN_HEIGHT // 2
        #Clamp camera to world bounds
        camera_x = max(0, min(self.screen_width - self.screen_width, camera_x))
        camera_y = max(0, min(self.screen_height - self.screen_height, camera_y))
        
        #Draw player
        self.player.draw(self.screen, camera_x, camera_y)
        
        #Draw enemies
        for enemy in self.enemies:
            enemy.draw(self.screen, camera_x, camera_y)
        
        #Draw bullets
        for bullet in self.bullets:
            bullet.draw(self.screen, camera_x, camera_y)

        pygame.display.flip()
    def close(self):
        if self.screen is not None:

            pygame.display.quit()
            pygame.quit()
            self.isopen = False