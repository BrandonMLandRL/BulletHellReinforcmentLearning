from gymnasium.envs.registration import register


def register_envs() -> None:
    """Register the Bullet Hell environment with Gymnasium."""
    register(
        id="BulletHell-v0",
        entry_point="bulllet_hell_rl.envs.BulletHellEnv:BulletHellEnv",
        max_episode_steps=1000,
    )


# If you want to test registration manually, you can run this module directly.
# However, in normal usage Gymnasium will discover this via the
# `gymnasium.envs` entry-point defined in `pyproject.toml`.
# if __name__ == "__main__":
#     import gymnasium as gym
#     register_envs()
#     env = gym.make("BulletHell-v0")