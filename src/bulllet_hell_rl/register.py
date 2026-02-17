from gymnasium.envs.registration import register

def register_envs():
    register(
        id="BulletHell-v0",
        entry_point="bullet_hell_rl.envs:BulletHellEnv",
        max_episode_steps=1000,
    )
# if __name__ == "__main__":
#     gym.register(id="BulletHellEnv-v0", 
#                  entry_point="bullet_hell_rl.envs.BulletHellEnv:BulletHellEnv",)