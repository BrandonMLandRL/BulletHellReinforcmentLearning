from envs.BulletHellEnv import BulletHellEnv
import gymnasium as gym
if __name__ == "__main__":
    # env = BulletHellEnv(render_mode="human")
    env = gym.make("BulletHellEnv-v0")
    for i in range(0,10):
        obs = env.reset()
        while True:
            # obs_space = env.observation_space
            # print(obs_space)
            action = env.action_space.sample()
            obs, rewards, done, info = env.step(action)
            env.render()

            if done:
                break