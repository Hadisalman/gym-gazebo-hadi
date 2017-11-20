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
import memory
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
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from IPython import embed

os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.log_device_placement=False
config.allow_soft_placement = True
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 1
sess = tf.Session(config=config)
K.set_session(sess)

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 1



parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='GazeboCircuit2cTurtlebotCameraNnEnv-v0')
parser.add_argument('--weights', type=str, default='dqn_GazeboCircuit2cTurtlebotCameraNnEnv-v0_weights.h5f')
args = parser.parse_args()

# Get the environment and extract the number of actions.
# env = gym.make(args.env_name)
env = gym.make('GazeboCircuit2cTurtlebotCameraNnEnv-v0')

np.random.seed(123)
env.seed(123)
#TODO modify the turtlebot environment so that it accomodates with gym environments (action_space ...)
nb_actions = 3 #env.action_space.n

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE


# obs,reward,done,_ =env.step(0)
# embed()

model = Sequential()
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering.')
model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
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
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=50000)]
    callbacks += [FileLogger(log_filename, interval=100)]

    #weights_filename = args.weights
    #dqn.load_weights(weights_filename)    
    
    dqn.fit(env, callbacks=callbacks, nb_steps=3000000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)
    
    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)

elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=False)



# if __name__ == '__main__':

#     #REMEMBER!: turtlebot_cnn_setup.bash must be executed.
#     env = gym.make('GazeboCircuit2cTurtlebotCameraNnEnv-v0')
#     outdir = 'gazebo_gym_experiments/'

#     continue_execution = False
#     #fill this if continue_execution=True
#     weights_path = '/tmp/turtle_c2c_dqn_ep200.h5' 
#     monitor_path = '/tmp/turtle_c2c_dqn_ep200'
#     params_json  = '/tmp/turtle_c2c_dqn_ep200.json'

#     img_rows, img_cols, img_channels = env.img_rows, env.img_cols, env.img_channels
#     epochs = 100000
#     steps = 1000

#     if not continue_execution:
#         minibatch_size = 32
#         learningRate = 1e-3#1e6
#         discountFactor = 0.95
#         network_outputs = 3
#         memorySize = 100000
#         learnStart = 10000 # timesteps to observe before training
#         EXPLORE = memorySize # frames over which to anneal epsilon
#         INITIAL_EPSILON = 1 # starting value of epsilon
#         FINAL_EPSILON = 0.01 # final value of epsilon
#         explorationRate = INITIAL_EPSILON
#         current_epoch = 0
#         stepCounter = 0
#         loadsim_seconds = 0

#         deepQ = DeepQ(network_outputs, memorySize, discountFactor, learningRate, learnStart)
#         deepQ.initNetworks()
#         # env.monitor.start(outdir, force=True, seed=None)
#         env = gym.wrappers.Monitor(env, directory=outdir, force=True, write_upon_reset=True)

#     else:
#         #Load weights, monitor info and parameter info.
#         with open(params_json) as outfile:
#             d = json.load(outfile)
#             explorationRate = d.get('explorationRate')
#             minibatch_size = d.get('minibatch_size')
#             learnStart = d.get('learnStart')
#             learningRate = d.get('learningRate')
#             discountFactor = d.get('discountFactor')
#             memorySize = d.get('memorySize')
#             network_outputs = d.get('network_outputs')
#             current_epoch = d.get('current_epoch')
#             stepCounter = d.get('stepCounter')
#             EXPLORE = d.get('EXPLORE')
#             INITIAL_EPSILON = d.get('INITIAL_EPSILON')
#             FINAL_EPSILON = d.get('FINAL_EPSILON')
#             loadsim_seconds = d.get('loadsim_seconds')

#         deepQ = DeepQ(network_outputs, memorySize, discountFactor, learningRate, learnStart)
#         deepQ.initNetworks()
#         deepQ.loadWeights(weights_path)

#         clear_monitor_files(outdir)
#         copy_tree(monitor_path,outdir)
#         # env.monitor.start(outdir, resume=True, seed=None)

#     last100Rewards = [0] * 100
#     last100RewardsIndex = 0
#     last100Filled = False

#     start_time = time.time()

#     #start iterating from 'current epoch'.
#     for epoch in xrange(current_epoch+1, epochs+1, 1):
#         observation = env.reset()
#         cumulated_reward = 0

#         # number of timesteps
#         for t in xrange(steps):
#             qValues = deepQ.getQValues(observation)

#             action = deepQ.selectAction(qValues, explorationRate)
#             newObservation, reward, done, info = env.step(action)

#             deepQ.addMemory(observation, action, reward, newObservation, done)
#             observation = newObservation

#             #We reduced the epsilon gradually
#             if explorationRate > FINAL_EPSILON and stepCounter > learnStart:
#                 explorationRate -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

#             if stepCounter == learnStart:
#                 print("Starting learning")

#             if stepCounter >= learnStart:
#                 deepQ.learnOnMiniBatch(minibatch_size, False)

#             if (t == steps-1):
#                 print ("reached the end")
#                 done = True

#             # env.monitor.flush(force=True)
#             cumulated_reward += reward

#             if done:
#                 last100Rewards[last100RewardsIndex] = cumulated_reward
#                 last100RewardsIndex += 1
#                 if last100RewardsIndex >= 100:
#                     last100Filled = True
#                     last100RewardsIndex = 0
#                 m, s = divmod(int(time.time() - start_time + loadsim_seconds), 60)
#                 h, m = divmod(m, 60)
#                 if not last100Filled:
#                     print ("EP "+str(epoch)+" - {} steps".format(t+1)+" - CReward: "+str(round(cumulated_reward, 2))+"  Eps="+str(round(explorationRate, 2))+"  Time: %d:%02d:%02d" % (h, m, s))
#                 else :
#                     print ("EP "+str(epoch)+" - {} steps".format(t+1)+" - last100 C_Rewards : "+str(int((sum(last100Rewards)/len(last100Rewards))))+" - CReward: "+str(round(cumulated_reward, 2))+"  Eps="+str(round(explorationRate, 2))+"  Time: %d:%02d:%02d" % (h, m, s))
#                     #SAVE SIMULATION DATA
#                     if (epoch)%100==0: 
#                         #save model weights and monitoring data every 100 epochs. 
#                         deepQ.saveModel(outdir+'turtle_c2c_dqn_ep'+str(epoch)+'.h5')
#                         # env.monitor.flush()
#                         copy_tree(outdir,'turtle_c2c_dqn_ep'+str(epoch))
#                         #save simulation parameters.
#                         parameter_keys = ['explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_outputs','current_epoch','stepCounter','EXPLORE','INITIAL_EPSILON','FINAL_EPSILON','loadsim_seconds']
#                         parameter_values = [explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_outputs, epoch, stepCounter, EXPLORE, INITIAL_EPSILON, FINAL_EPSILON,s]
#                         parameter_dictionary = dict(zip(parameter_keys, parameter_values))
#                         with open(outdir+'turtle_c2c_dqn_ep'+str(epoch)+'.json', 'w') as outfile:
#                             json.dump(parameter_dictionary, outfile)
#                 break

#             stepCounter += 1
#             if stepCounter % 2500 == 0:
#                 print("Frames = "+str(stepCounter))

#     env.monitor.close() #not needed in latest gym update
#     env.close()
