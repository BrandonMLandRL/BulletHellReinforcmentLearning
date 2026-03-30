# -*- coding: utf-8 -*-
"""
Deep Learning Reinforcement Tutorial: Deep Q Network (DQN) = Combination of Deep Learning and Q-Learning Tutorial

The class developed in this file implements the Deep Q Network (DQN) Reinforcement Learning Algorithm.
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
# import the necessary libraries
import numpy as np
import socket
from bullet_hell_rl.net import protocol
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from collections import deque 
from tensorflow import gather_nd
from tensorflow.keras.losses import MSE as mean_squared_error 
import threading
import queue

class Actor:
#Actor will also be able to establish a connection to localhost port 5556 this will be the learner.     
    ###########################################################################
    #   START - __init__ function
    ###########################################################################
    # INPUTS: 
    # env - Cart Pole environment
    # gamma - discount rate
    # epsilon - parameter for epsilon-greedy approach
    # numberEpisodes - total number of simulation episodes
    
            
    def __init__(self, on_message_callback=None):       
        self.epsilon=1
        self.learner_socket = None
        # state dimension
        self.stateDimension=63
        # action dimension (5 moves × 4 aim angles at 90° steps = 20)
        self.actionDimension=20
        
        #mainNetwork will be created by loading in the .h5 weights file (or future better models will have .keras)
        self.mainNetwork=None

        self.index = 0 
        #Index and epsilon will be updated via message by the learner when it broadcasts weights.
        #Or when the learner sends out its init packet. 
 # Queues for cross-thread communication
        self._recv_queue = queue.Queue()   # messages from learner -> actor parent
        self._send_queue = queue.Queue()   # experience tuples from actor -> learner
        self._stop_event = threading.Event()
        self._recv_thread = None
        self._send_thread = None
        self._on_message = on_message_callback  # optional callback parent can pass in
        # Connect to learner
        try:
            self.learner_socket = socket.create_connection(("127.0.0.1", 5556), timeout=2.0)
            print("Connected to Learner on 127.0.0.1:5556")
            self.learner_socket.settimeout(None)
            hello = protocol.recv_message(self.learner_socket)
            if hello and hello.get("type") == protocol.MSG_WELCOME:
                print(f"Learner says: {hello.get('message', '')}")
            # Start background threads once connection is ready
            self._start_background_threads()
        except OSError:
            self.learner_socket = None
            print("Failed to connect to TCP Learner")
    def _start_background_threads(self):
        if self.learner_socket is None:
            return
        self._recv_thread = threading.Thread(
            target=self._actor_recv_thread,
            name="ActorRecv",
            daemon=True,
        )
        self._recv_thread.start()
        self._send_thread = threading.Thread(
            target=self._actor_send_thread,
            name="ActorSend",
            daemon=True,
        )
        self._send_thread.start()
    
    def _actor_recv_thread(self):
        """Background thread that continuously receives messages from the learner."""
        sock = self.learner_socket
        if sock is None:
            return
        while not self._stop_event.is_set():
            msg = protocol.recv_message(sock)
            if msg is None:
                # Connection closed or error
                break
            # Push into the receive queue
            self._recv_queue.put(msg)
            # Optionally notify parent immediately
            if self._on_message is not None:
                try:
                    self._on_message(msg)
                except Exception as e:
                    # Avoid killing the thread from user callback errors
                    print(f"Actor on_message callback error: {e}")
        # Clean up socket on exit
        try:
            sock.close()
        except OSError:
            pass
        self.learner_socket = None

    def _actor_send_thread(self):
        """Background thread that sends messages enqueued in _send_queue to learner."""
        sock = self.learner_socket
        if sock is None:
            return
        while not self._stop_event.is_set():
            try:
                msg = self._send_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg is None:
                # Sentinel to signal shutdown
                break
            try:
                protocol.send_message(sock, msg)
            except OSError:
                # Socket died; stop this thread
                break
        try:
            sock.close()
        except OSError:
            pass
        self.learner_socket = None

    def close(self):
        """Stop background threads and close connection."""
        self._stop_event.set()
        # Wake send thread
        self._send_queue.put(None)

        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=1.0)
        if self._send_thread and self._send_thread.is_alive():
            self._send_thread.join(timeout=1.0)

        if self.learner_socket is not None:
            try:
                self.learner_socket.close()
            except OSError:
                pass
            self.learner_socket = None

    def my_loss_fn(self,y_true, y_pred):
        
        s1,_=y_true.shape
        #print(s1,s2)
        
        # this matrix defines indices of a set of entries that we want to 
        # extract from y_true and y_pred
        # s2=2
        # s1=self.batchReplayBufferSize
        indices=np.zeros(shape=(s1,2))
        indices[:,0]=np.arange(s1)
        indices[:,1]=self.actionsAppend
        
        # gather_nd and mean_squared_error are TensorFlow functions
        loss = mean_squared_error(gather_nd(y_true,indices=indices.astype(int)), gather_nd(y_pred,indices=indices.astype(int)))
        #print(loss)
        return loss    
    ###########################################################################
    #   END - of function my_loss_fn
    ###########################################################################
    
    
    ###########################################################################
    #   START - helper functions for BulletHellEnv
    ###########################################################################

    # def _flatten_obs(self, obs):
    #     """
    #     Flatten BulletHellEnv dict observation into a 1D vector of length self.stateDimension.
    #     """
    #     player = obs["player"].ravel()
    #     enemies = obs["enemies"].ravel()
    #     bullets = obs["bullets"].ravel()
    #     return np.concatenate([player, enemies, bullets], axis=0)

    # def _flat_action_to_env(self, flat_action):
    #     """
    #     Map flat action index in [0, 20] to BulletHellEnv dict action.
    #     Move: 0-4. Aim: 0, 90, 180, 360 (4 choices).
    #     """
        
    #     move = int(flat_action // 4)
    #     angle = int(flat_action % 4) * 90
    #     fire_angle = np.array([angle], dtype=np.int16)
    #     return {"move": move, "fire_angle": fire_angle}

    ###########################################################################
    #   END - helper functions for BulletHellEnv
    ###########################################################################
       
    ###########################################################################
    #    START - function for selecting an action: epsilon-greedy approach
    ###########################################################################
    # this function selects an action on the basis of the current state 
    # INPUTS: 
    # state - state for which to compute the action
    # index - index of the current episode
    def selectAction(self,state,index):        
        # first index episodes we select completely random actions to have enough exploration
        # change this
        if index<1:
            return np.random.choice(self.actionDimension)   
            
        # Returns a random real number in the half-open interval [0.0, 1.0)
        # this number is used for the epsilon greedy approach
        randomNumber=np.random.random()
        
        # after index episodes, we slowly start to decrease the epsilon parameter
        if index>200:
            self.epsilon=0.999*self.epsilon
        
        # if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber < self.epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.actionDimension)            
        # otherwise, we are selecting greedy actions
        else:
            # we return the index where Qvalues[state,:] has the max value
            # that is, since the index denotes an action, we select greedy actions
            # print("from learning")
            Qvalues=self.mainNetwork.predict(state.reshape(1,self.stateDimension), verbose=0)
          
            return np.random.choice(np.where(Qvalues[0,:]==np.max(Qvalues[0,:]))[0])
            # here we need to return the minimum index since it can happen
            # that there are several identical maximal entries, for example 
            # import numpy as np
            # a=[0,1,1,0]
            # np.where(a==np.max(a))
            # this will return [1,2], but we only need a single index
            # that is why we need to have np.random.choice(np.where(a==np.max(a))[0])
            # note that zero has to be added here since np.where() returns a tuple
    ###########################################################################
    #    END - function selecting an action: epsilon-greedy approach
    ###########################################################################
    

            
            
            
            
            
            
            
            
        
        
        
        
        
        
    