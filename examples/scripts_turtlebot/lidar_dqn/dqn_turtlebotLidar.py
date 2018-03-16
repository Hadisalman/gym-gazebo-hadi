#!/usr/bin/env python

'''
Based on: 
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
from __future__ import division
import gym
import cv2
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.initializers import normal
from keras import optimizers
from keras.optimizers import RMSprop
from keras.layers import Convolution2D, Flatten, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import SGD , Adam
from keras import backend
import keras.backend as K
import tensorflow as tf

import argparse
from PIL import Image
import numpy as np
import gym

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, tensorboardLogger
from IPython import embed

os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.log_device_placement=False
config.allow_soft_placement = True
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 1
sess = tf.Session(config=config)
K.set_session(sess)

save_dir = 'dqn_lidar_weights/'

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='GazeboMax1TurtlebotLidar-v0')
parser.add_argument('--continue-training', action='store_true', 
        help='Flag whether to load check point and continue training')
parser.add_argument('--weights', type=str, default=save_dir+'dqn_GazeboCircuit2TurtlebotLidarNn-v0_weights_3000000.h5f',
        help='Weights file to use during training (with --continue-training flag on) or during testing')
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)

#TODO modify the turtlebot environment so that it accomodates with gym environments (action_space ...)
nb_actions = env.action_space.n
WINDOW_LENGTH=1
# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
# input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

# obs,reward,done,_ =env.step(0)
# embed()

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

# class turtleBotProcessor(Processor):
# processor = turtleBotProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.00,
                              nb_steps=100000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                nb_steps_warmup=5000, gamma=.99, target_model_update=1000,
               enable_dueling_network=True, dueling_type='avg', train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

log_parent_dir = './train_log'

def make_log_dir():
    import datetime, os
    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(log_parent_dir, args.env_name, current_timestamp)
    os.makedirs(log_dir)
    return log_dir

if args.mode == 'train':
    log_dir = make_log_dir()     

    checkpoint_weights_filename = os.path.join(log_dir, '{step}.h5f')
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=50000)]
    callbacks += [tensorboardLogger(log_dir)]

    # log_filename = 'dqn_{}_log.json'.format(args.env_name)
    # callbacks += [FileLogger(log_filename, interval=100)]

    if args.continue_training:
        weights_filename = args.weights
        dqn.load_weights(weights_filename)  

    dqn.fit(env, callbacks=callbacks, nb_steps=300000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    weights_filename =os.path.join(log_dir, 'dqn_{}_weights.h5f'.format(args.env_name))
    dqn.save_weights(weights_filename, overwrite=True)
    
    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)

elif args.mode == 'test':

    if args.weights:
        weights_filename = args.weights
    else:
        raise "Please specify the path to the weights file"
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=False)
