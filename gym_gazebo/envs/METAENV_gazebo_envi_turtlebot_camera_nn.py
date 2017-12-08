import cv2
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
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from gym.utils import seeding
from cv_bridge import CvBridge, CvBridgeError

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
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

        # self.last50actions = [0] * 50

        self.img_rows = 84
        self.img_cols = 84
        self.img_channels = 1
        self.initial_angles = [np.pi/2,np.pi/2]

        
        self.weight_file_1 = '/home/hadis/Hadi/RL/gym-gazebo-hadi/examples/scripts_turtlebot/camera_dqn/weights_to_use_DL/path1.h5f'
        self.weight_file_2 = '/home/hadis/Hadi/RL/gym-gazebo-hadi/examples/scripts_turtlebot/camera_dqn/weights_to_use_DL/path2.h5f'
        
        memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.2, value_min=.1, value_test=.05,
                              nb_steps=100000)

        self.model1 = self.create_model(self.weight_file_1)
        self.model2 = self.create_model(self.weight_file_2)
        
        self.dqn1 = DQNAgent(self.model1, nb_actions=3, policy=policy, memory=memory,
                nb_steps_warmup=10000, gamma=.99, target_model_update=1000,
               enable_dueling_network=True, dueling_type='avg', train_interval=4)
        self.dqn2 = DQNAgent(self.model2, nb_actions=3, policy=policy, memory=memory,
                nb_steps_warmup=10000, gamma=.99, target_model_update=1000,
               enable_dueling_network=True, dueling_type='avg', train_interval=4)
        
        self.dqn1.compile(Adam(lr=.00025), metrics=['mae'])
        self.dqn2.compile(Adam(lr=.00025), metrics=['mae'])

        self.dqn1.load_weights(self.weight_file_1)
        self.dqn2.load_weights(self.weight_file_2)
        
        self.current_state = 0

        self.frame_buffer = np.zeros((WINDOW_LENGTH,self.img_rows,self.img_cols))
# Set up the code for: 
    # Tensorboard Which policy is active
    # Supervision
    # Running ground truth policy.     

    def create_model(self, weight_file):
        input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
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
        nb_actions = 3
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))

        # model.compile(RMSprop(lr=.000025), loss)
        # if not(weight_file==''):
        #     model.load_weights(weight_file)            
        
        return model

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

    def calculate_observation(self,data):
        min_range = 0.21
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        return done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Step now runs a meta step.
    def _step(self, meta_action):
        # embed()
        print("CALLING META STEP")
        number_steps = 5
        is_done = False
        cummulative_reward = 0.        
        k=0
        # For how many ever steps the low level policy is to be executed
        while not(is_done) and (k<number_steps):
            meta_action=0

            current_state = getModelStates.gms_client('mobile_base','world')

            if (current_state.pose.position.y > 7.0):
                meta_action=1
            if (current_state.pose.position.y > 7.0 and current_state.pose.position.x<-1):
                meta_action=0
                
            print("Meta action:",meta_action)
            if meta_action==0:
                # lowlevel_action = np.argmax(self.model1.predict(self.frame_buffer.reshape(1,4,84,84)))
                lowlevel_action = self.dqn1.forward(self.current_state)
                self.current_state, onestep_reward, is_done, _ = self.lowerlevel_step(lowlevel_action)
                # observation, r, d, info = env.step(action)
                # observation = deepcopy(observation)
                self.dqn1.backward(onestep_reward, terminal=is_done)

            if meta_action==1:
                # lowlevel_action = np.argmax(self.model2.predict(self.frame_buffer.reshape(1,4,84,84)))
                lowlevel_action = self.dqn2.forward(self.current_state)
                self.current_state, onestep_reward, is_done, _ = self.lowerlevel_step(lowlevel_action)
                # observation, r, d, info = env.step(action)
                # observation = deepcopy(observation)
                self.dqn2.backward(onestep_reward, terminal=is_done)

            # print("Low level",lowlevel_action)

            
            # print("IS IT DONE?", is_done)
            # Adding the one step reward to the cummulative_reward.
            cummulative_reward += onestep_reward
            k+=1
                
            np.roll(self.frame_buffer,-1,axis=0)
            self.frame_buffer[WINDOW_LENGTH-1] = copy.deepcopy(self.current_state)

        return self.current_state, cummulative_reward, is_done, _

    # def _step(self, action):
    def lowerlevel_step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException, e:
            print ("/gazebo/unpause_physics service call failed")

        '''# 21 actions
        max_ang_speed = 0.3
        ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.2
        vel_cmd.angular.z = ang_vel
        self.vel_pub.publish(vel_cmd)'''

        # 3 actions

        # REMOVE THIS LATAERRR
        # print("FIXING ACTION TO FORWARD TEMPORARILY")
        # action = 0 
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

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        done = self.calculate_observation(data)

        image_data = None
        success=False
        cv_image = None
        while image_data is None or success is False:
            try:
                image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                #temporal fix, check image is not corrupted
                if not (cv_image[h/2,w/2,0]==178 and cv_image[h/2,w/2,1]==178 and cv_image[h/2,w/2,2]==178):
                    success = True
                else:
                    pass
                    #print("/camera/rgb/image_raw ERROR, retrying")
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")


        self.last50actions.pop(0) #remove oldest
        if action == 0:
            self.last50actions.append(0)
        else:
            self.last50actions.append(1)

        action_sum = sum(self.last50actions)


        '''# 21 actions
        if not done:
            # Straight reward = 5, Max angle reward = 0.5
            reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))
        
            if action_sum > 45: #L or R looping
                #print("90 percent of the last 50 actions were turns. LOOPING")
                reward = -5
        else:
            reward = -200'''


        # Add center of the track reward
        # len(data.ranges) = 100
        laser_len = len(data.ranges)
        left_sum = sum(data.ranges[laser_len-(laser_len/5):laser_len-(laser_len/10)]) #80-90
        right_sum = sum(data.ranges[(laser_len/10):(laser_len/5)]) #10-20

        center_detour = abs(right_sum - left_sum)/5

        # 3 actions
        # if not done:
        #     if action == 0:
        #         reward = 10 / float(center_detour+1)
        #     elif action_sum > 45: #L or R looping
        #         reward = -0.5
        #     else: #L or R no looping
        #         reward = 5 / float(center_detour+1)
        # else:
        #     # reward = -1
        #     reward = -50

        if not done:
            if action == 0:
                reward = 5
            else:
                reward = 1
        else:
            reward = -200

        #print("detour= "+str(center_detour)+" :: reward= "+str(reward)+" ::action="+str(action))

        '''x_t = skimage.color.rgb2gray(cv_image)
        x_t = skimage.transform.resize(x_t,(32,32))
        x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))'''
        
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        #cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
        #cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))

        state = cv_image.reshape(cv_image.shape[0], cv_image.shape[1])
        return state, reward, done, {}

        # test STACK 4
        #cv_image = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        #self.s_t = np.append(cv_image, self.s_t[:, :3, :, :], axis=1)
        #return self.s_t, reward, done, {} # observation, reward, done, info

    def _reset(self):
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

        image_data = None
        success=False
        cv_image = None
        while image_data is None or success is False:
            try:
                image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                #temporal fix, check image is not corrupted
                if not (cv_image[h/2,w/2,0]==178 and cv_image[h/2,w/2,1]==178 and cv_image[h/2,w/2,2]==178):
                    success = True
                else:
                    pass
                    #print("/camera/rgb/image_raw ERROR, retrying")
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")

        '''x_t = skimage.color.rgb2gray(cv_image)
        x_t = skimage.transform.resize(x_t,(32,32))
        x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))'''


        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        #cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
        #cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))

        state = cv_image.reshape(cv_image.shape[0], cv_image.shape[1])

        self.current_state = state

        for i in range(WINDOW_LENGTH):
            self.frame_buffer[i,:,:] = state

        return state

        # test STACK 4
        #self.s_t = np.stack((cv_image, cv_image, cv_image, cv_image), axis=0)
        #self.s_t = self.s_t.reshape(1, self.s_t.shape[0], self.s_t.shape[1], self.s_t.shape[2])
        #return self.s_t
