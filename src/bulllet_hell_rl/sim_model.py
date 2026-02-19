"""
Deep Learning Reinforcement Tutorial: Deep Q Network (DQN) = Combination of Deep Learning and Q-Learning Tutorial

This file creates a movie that shows the performance of the trained model

The webpage explaining the codes and the main idea of the DQN is given here:

https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/


Author: Aleksandar Haber 
Date: February 2023

Tested on:

tensorboard==2.11.2
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow==2.11.0
tensorflow-estimator==2.11.0
tensorflow-intel==2.11.0
tensorflow-io-gcs-filesystem==0.30.0

keras==2.11.0

gym==0.26.2

"""
# you will also need to install MoviePy, and you do not need to import it explicitly
# pip install moviepy

# import Keras
import keras

# import the class

# import gym
import gymnasium as gym

# numpy
import numpy as np

from bulllet_hell_rl.DQN.DQNLegacy import DeepQLearning

from bulllet_hell_rl.envs.BulletHellEnv import BulletHellEnv

# create the environment
env = BulletHellEnv(render_mode="human")

# DQN instance provides _flatten_obs, _flat_action_to_env, and stateDimension
dqn = DeepQLearning(env, gamma=0.99, epsilon=0.0, numberEpisodes=1, modelFileName="")
# load trained weights into main network
dqn.mainNetwork = keras.models.load_model(
    "centerlower.h5", custom_objects={"my_loss_fn": DeepQLearning.my_loss_fn}
)

# number of episodes to run (example evaluation)
numberEpisodes = 5
sumRewardsEpisode = []

# Wrapper for recording the video (optional)
# video_length = 400
# env = gym.wrappers.RecordVideo(env, 'stored_video', video_length=video_length)

for indexEpisode in range(numberEpisodes):
    rewardsEpisode = []

    print(f"Simulating episode {indexEpisode} out of {numberEpisodes}")

    # reset the environment at the beginning of every episode
    (currentState, _) = env.reset()
    currentState = dqn._flatten_obs(currentState)

    terminalState = False
    while not terminalState:
        # greedy action from Q-values (no exploration)
        Qvalues = dqn.mainNetwork.predict(
            currentState.reshape(1, dqn.stateDimension), verbose=0
        )

        action = np.random.choice(
            np.where(Qvalues[0, :] == np.max(Qvalues[0, :]))[0]
        )

        env_action = dqn._flat_action_to_env(action)
        (nextState, reward, terminated, truncated, _) = env.step(env_action)
        terminalState = terminated or truncated
        nextState = dqn._flatten_obs(nextState)
        rewardsEpisode.append(reward)

        currentState = nextState

    print("Sum of rewards {}".format(np.sum(rewardsEpisode)))
    sumRewardsEpisode.append(np.sum(rewardsEpisode))

env.close()



