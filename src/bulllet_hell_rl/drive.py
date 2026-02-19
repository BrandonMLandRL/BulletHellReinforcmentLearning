"""
Deep Learning Reinforcement Tutorial: Deep Q Network (DQN) = Combination of Deep Learning and Q-Learning Tutorial

This file contains driver code that imports DeepQLearning class developed in the file "functions_final"
 
The class DeepQLearning implements the Deep Q Network (DQN) Reinforcement Learning Algorithm.
The implementation is based on the OpenAI Gym Cart Pole environment and TensorFlow (Keras) machine learning library

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

import keras 

from DQN.DQNLegacy import DeepQLearning

import gymnasium as gym
from envs.BulletHellEnv import BulletHellEnv

# arg management
import sys

# for saving and loading of supplementary parameters
import pickle

# detect len of args, if arg length != 2 throw incorrect argument count error 
if len(sys.argv) == 2: # the script is run with no arguments

    # check if argv[1] == string w .keras extension
    filename = sys.argv[1] if (".h5" in str(sys.argv[1])) else None
    if filename == None:
        raise Exception("Incorrect script argument. Provide a filename ending in .h5")

    # create environment
    # env=gym.make('CartPole-v1')
    env = BulletHellEnv(render_mode="human")

    # select the parameters
    gamma=.99
    # probability parameter for the epsilon-greedy approach
    epsilon=1
    # number of training episodes
    # NOTE HERE THAT AFTER CERTAIN NUMBERS OF EPISODES, WHEN THE PARAMTERS ARE LEARNED
    # THE EPISODE WILL BE LONG, AT THAT POINT YOU CAN STOP THE TRAINING PROCESS BY PRESSING CTRL+C
    # DO NOT WORRY, THE PARAMETERS WILL BE MEMORIZED
    numberEpisodes=100000
    try:
        # model_exists = False
        # # check if the model already exists
        # try:
        #     with open(f"{filename}.pkl", 'rb') as pkl_file:
        #         ld = pickle.load(pkl_file)a
        #     print(f"Loaded data: {ld}")
        #     model_exists = True
        # except FileNotFoundError:
        #     print(f"Error: The file {filename} was not found.")
        # except pickle.UnpicklingError:
        #     print(f"Error: Could not unpickle data from the file {filename}.")
        #     # check if the pair save file exists

        #     # if so, load parameters of the model.  
        # if model_exists:
        #     # incorporate the loaded data (ld) into the model 
        #     LearningQDeep = DeepQLearning(env, ld["gamma"], ld["epsilon"], numberEpisodes-ld["episodeIndex"], filename)
        #     loaded_model = keras.models.load_model("my_model.h5",custom_objects={'my_loss_fn':DeepQLearning.my_loss_fn})
        #     print(f"HEERREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE    {type(loaded_model)}")
        #     LearningQDeep.mainNetwork = loaded_model
        #     # LearningQDeep.gamma = ld["gamma"]
        #     # LearningQDeep.epsilon = ld["epsilon"]
        #     # LearningQDeep.numberEpisodes = numberEpisodes-ld["episodeIndex"]

        # else: 
        LearningQDeep=DeepQLearning(env,gamma,epsilon,numberEpisodes, filename)
        
        # run the learning process
        LearningQDeep.trainingEpisodes()
        # get the obtained rewards in every episode
        LearningQDeep.sumRewardsEpisode

        #  summarize the model
        LearningQDeep.mainNetwork.summary()
        # save the model, this is important, since it takes long time to train the model 
        # and we will need model in another file to visualize the trained model performance
        LearningQDeep.mainNetwork.save("trained_model_temp.h5")
    except KeyboardInterrupt:
        # after keyboard interrupt, save all necessary items that are required to resume model training where it was. 
        # parameters that change over time include the 

        # current episode index
        episodeIndex = LearningQDeep.episodeIndex

        # replayBuffer which is a deque
        replayBuffer = LearningQDeep.replayBuffer
        # epsilon 

        epsilon = LearningQDeep.epsilon
        # gamma 
        gamma = LearningQDeep.gamma
        try:
            with open(f"{filename}.pkl", "wb") as f:
                data_dict = {
                    "episodeIndex" : episodeIndex,
                    "replayBuffer" : replayBuffer,
                    "epsilon" : epsilon,
                    "gamma" : gamma,
                }
                
                pickle.dump(data_dict, f)
            print(f"Successfully pickled dictionary to file: {filename}.pkl")
        except IOError as e:
            print(f"Error saving file: {e}")


else:
    raise Exception("Please provide a file name ending in .h5 for the model to be saved or loaded from")


