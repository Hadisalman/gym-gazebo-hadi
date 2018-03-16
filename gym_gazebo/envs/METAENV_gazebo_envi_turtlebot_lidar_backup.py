import gym
import rospy
import roslaunch
import time
import numpy as np
import sys
import os
import random
import copy
import roslib; roslib.load_manifest('gazebo_ros')

from gazebo_msgs.msg import ModelState
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from gym.utils import seeding

from IPython import embed
from tf.transformations import quaternion_from_euler

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

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, tensorboardLogger

import getModelStates


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
nb_actions = 2 #env.action_space.n


class MetaGazeboEnviTurtlebotCameraNnEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "MetaGazeboEnviTurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.state_pub = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=10)

        self.reward_range = (-np.inf, np.inf)
        self.episode = 0
        self._seed()
        
        #Modify the Meta networks
        self.scan_size = 100

        currentPath = os.getcwd()
        self.weight_file_1 = currentPath + '/weights_to_use_DL/path1.h5f'
        self.weight_file_2 = currentPath + '/weights_to_use_DL/path2.h5f'
        self.weight_file_3 = currentPath + '/weights_to_use_DL/path3.h5f'
        
        memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.2, value_min=.1, value_test=.05,
                              nb_steps=100000)

        # Create same lower level models
        self.model1 = self.create_model()
        self.model2 = self.create_model()
        self.model3 = self.create_model()
        
        self.dqn1 = DQNAgent(self.model1, nb_actions=3, policy=policy, memory=memory,
                             nb_steps_warmup=10000, gamma=.99, target_model_update=1000,
                             enable_dueling_network=True, dueling_type='avg', train_interval=4)
        self.dqn2 = DQNAgent(self.model2, nb_actions=3, policy=policy, memory=memory,
                             nb_steps_warmup=10000, gamma=.99, target_model_update=1000,
                             enable_dueling_network=True, dueling_type='avg', train_interval=4)
        self.dqn3 = DQNAgent(self.model3, nb_actions=3, policy=policy, memory=memory,
                             nb_steps_warmup=10000, gamma=.99, target_model_update=1000,
                             enable_dueling_network=True, dueling_type='avg', train_interval=4)
        
        self.dqn1.compile(Adam(lr=.00025), metrics=['mae'])
        self.dqn2.compile(Adam(lr=.00025), metrics=['mae'])
        self.dqn3.compile(Adam(lr=.00025), metrics=['mae'])
        
        self.dqn1.load_weights(self.weight_file_1)
        self.dqn2.load_weights(self.weight_file_2)
        self.dqn3.load_weights(self.weight_file_3)
        self.episode_reward_array = []
        self.episode_reward=0
        self.current_episode = []
        self.activation_history = []
        self.current_state = 0
        self.hand_crafte_policy = True
        self.frame_buffer = np.zeros((WINDOW_LENGTH,self.img_rows,self.img_cols))

    ######################################################################
    ##           Calculate observation based on Lidar inputs            ##
    ######################################################################
    def calculate_observation(self,data):
        min_range = 0.21
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        return state, done

    ## generate sample datas
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    ######################################################################
    ##                   Create the network model                       ##
    ######################################################################
    def create_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dense(env.action_space.n))
        model.add(Activation('linear'))
        return model

    ######################################################################
    ##               Output the action message                          ##
    ######################################################################    
    def setmodelstate(self, modelname='mobile_base',x=0,y=0,yaw=0):
        # rospy.init_node('ali')    
        state=ModelState()
        state.model_name = modelname
        state.pose.position.x=x
        state.pose.position.y=y
        yaw = np.pi/2
        q = quaternion_from_euler(0,0,yaw)
        state.pose.orientation.x = q[0]
        state.pose.orientation.y = q[1]
        state.pose.orientation.z = q[2]
        state.pose.orientation.w = q[3]
        self.state_pub.publish(state)

    # Step now runs a meta step.
    def _step(self, meta_action):
        # embed()
        print("CALLING META STEP")
        number_steps = 25
        is_done = False
        cummulative_reward = 0.        
        k=0
        # For how many ever steps the low level policy is to be executed
        while not(is_done) and (k<number_steps):
            current_state = getModelStates.gms_client('mobile_base','world')
            
            if self.hand_crafte_policy:          
                meta_action=0
                if (current_state.pose.position.y > 7.0):
                    meta_action=2
                if (current_state.pose.position.y > 7.0 and current_state.pose.position.x<-1):
                    meta_action=1
                if (current_state.pose.position.y > 7.0) and current_state.pose.position.x<-0.5):
                    meta_action=0
                
            print("Meta action:",meta_action)
            if meta_action==0:
                lowlevel_action = self.dqn1.forward(self.current_state)
                self.current_state, onestep_reward, is_done, _ = \
                    self.lowerlevel_step(lowlevel_action)
                self.dqn1.backward(onestep_reward, terminal=is_done)

            if meta_action==1:
                lowlevel_action = self.dqn2.forward(self.current_state)
                self.current_state, onestep_reward, is_done, _ = \
                    self.lowerlevel_step(lowlevel_action)
                self.dqn2.backward(onestep_reward, terminal=is_done)

            if meta_action==2:
                lowlevel_action = self.dqn3.forward(self.current_state)
                self.current_state, onestep_reward, is_done, _ = \
                    self.lowerlevel_step(lowlevel_action)
                self.dqn3.backward(onestep_reward, terminal=is_done)

            self.current_episode.append([current_state.pose.position.x, \
                                         current_state.pose.position.y, \
                                         meta_action])

            self.episode_reward+= onestep_reward
            if (current_state.pose.position.y < -3.0) or (current_state.pose.position.x < -9):
                is_done=True

            print("Low level",lowlevel_action)
            print("IS IT DONE?", is_done)
            
            # Adding the one step reward to the cummulative_reward.
            cummulative_reward += onestep_reward
            k+=1    
            np.roll(self.frame_buffer,-1,axis=0)
            self.frame_buffer[WINDOW_LENGTH-1] = copy.deepcopy(self.current_state)

        if is_done:
            self.episode_reward_array.append(self.episode_reward)
            self.episode_reward=0

        return self.current_state, cummulative_reward, is_done, _
    
    ######################################################################
    ##               Generate transform funciton with LiDAR inputs      ##
    ######################################################################
    def lowerlevel_step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException, e:
            print ("/gazebo/unpause_physics service call failed")
            
        # 3 actions
        if action == 0: #FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1: #LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = 0.2
            self.vel_pub.publish(vel_cmd)
        elif action == 2: #RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = -0.2
            self.vel_pub.publish(vel_cmd)

        # caculate the lidar inputs
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")

        lidar_state, done = self.calculate_observation(data)
            
        self.last50actions.pop(0) #remove oldest
        if action == 0:
            self.last50actions.append(0)
        else:
            self.last50actions.append(1)

        action_sum = sum(self.last50actions)

        # Add center of the track reward
        laser_len = len(data.ranges)
        left_sum = sum(data.ranges[laser_len-(laser_len/5):laser_len-(laser_len/10)]) #80-90
        right_sum = sum(data.ranges[(laser_len/10):(laser_len/5)]) #10-20

        center_detour = abs(right_sum - left_sum)/5

        if not done:
            if action == 0:
                reward = 5
            else:
                reward = 1
        else:
            reward = -200

        #TODO generate state from LiDAR scan
        return state, reward, done, {}

    ######################################################################
    ##                      Reset the environments                      ##
    ##                      Use original Reset function                 ##
    ######################################################################
    def _reset(self):
        
        #SAvinf previous episode
        self.activation_history.append(self.current_episode)
        np.save("Activation_History.npy",self.activation_history)
        self.current_episode = []
        self.episode +=1 
        self.last50actions = [0] * 50 #used for looping avoidance
        
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
            self.setmodelstate(x=0,y=-2,yaw=self.initial_angles[self.episode%2])
        except rospy.ServiceException, e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except rospy.ServiceException, e:
            print ("/gazebo/unpause_physics service call failed")
        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")

        state,done = self.calculate_observation(data)

        return np.asarray(state)
