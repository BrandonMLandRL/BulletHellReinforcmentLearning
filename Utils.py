from BulletHellEnv import BulletHellEnv
import numpy as np

def get_state_dim():
    env = BulletHellEnv()
    obs_space = env.observation_space

    print(obs_space)

    state_dim = 0
    temp_list = [i for i in obs_space.values()]
    print(temp_list)
    for i in temp_list:
        print(i.shape)
        prod = 1
        for j in i.shape:
            prod = prod * j
        state_dim += prod

    print(state_dim)
    
    return state_dim


def get_action_dim():
    env = BulletHellEnv()
    action_space = env.action_space

    print(action_space)           

    action_dim = action_space['fire_angle'].high[0] * 5 

    print(action_dim)

if __name__ == "__main__":
    # get_state_dim()
    get_action_dim()