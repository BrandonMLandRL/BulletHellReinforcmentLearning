from bulllet_hell_rl.envs.BulletHellEnv import BulletHellEnv
import gymnasium as gym


if __name__ == "__main__":
    # Direct instantiation (no registry):
    # env = BulletHellEnv(render_mode="human")

    # Via Gymnasium registry (recommended once installed):
    # Ensure `register_envs()` has been called or the package entry point is loaded.
    env = gym.make("BulletHell-v0")

    for _ in range(10):
        obs, info = env.reset()
        while True:
            # obs_space = env.observation_space
            # print(obs_space)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if terminated or truncated:
                break