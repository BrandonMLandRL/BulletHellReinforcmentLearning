from BulletHellEnv import BulletHellEnv
import numpy as np

def get_dim():
    env = BulletHellEnv()
    obs_space = env.observation_space
    action_space = env.action_space
    #Get each of the state shapes 
    state_dim = 0

    

    print(state_dim)
    


if __name__ == "__main__":
    get_dim()