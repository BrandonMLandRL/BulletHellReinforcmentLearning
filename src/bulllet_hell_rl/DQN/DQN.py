#This file will create the DQN that will learn in the BulletHellEnv


#Dict('bullets': Box(-1.0, 1.0, (10, 4), float32), 'enemies': Box(-1.0, 1.0, (5, 4), float32), 'player': Box(0.0, 1.0, (3,), float32))
#Observation space printed: Bullets(N_bullets,4), Enemies (N_enemies,4), Player(3,)

#First we will collect the 

from src.bulllet_hell_rl.Utils import get_action_dim, get_state_dim


import tensorflow as tf
import tf_agents
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks.q_network import QNetwork
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
import numpy as np
import matplotlib.pyplot as plt

class DeepQNetwork:
    def __init__(self):
        #Mathy params

        self.action_dim = get_action_dim()
        self.obs_dim = get_state_dim()

        
if __name__ == "__main__":
    print("Creating DQN")
    dqn = DeepQNetwork()
    print("MUAHAHAHA")
