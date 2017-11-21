import gym
import rospy
import roslaunch
import time
import numpy as np
import cv2
import sys
import os
import random

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

import tools
from Functions.Controller import Controller
from Functions.CPGgs import CPGgs
from std_srvs.srv import Empty
from std_msgs.msg import Float64


class GazeboCircuit2cSnakeMonsterCameraNnEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2cSnakeMonsterLidar_v0.launch")
        # self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.action_space = spaces.Discrete(3) #F,L,R
        self.reward_range = (-np.inf, np.inf)

        self.pub={}
        
        for i in xrange(6) :
            for j in xrange(3) :
                self.pub['L' + str(i+1) + '_' + str(j+1)] = rospy.Publisher( '/snake_monster' + '/' + 'L' + str(i+1) + '_' + str(j+1) + '_'
                                            + 'eff_pos_controller' + '/command',
                                            Float64, queue_size=10 )

        self._seed()
        
        self.last50actions = [0] * 50
        self.img_rows = 84
        self.img_cols = 84
        self.img_channels = 1

    def calculate_observation(self,data):
        min_range = 0.4
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        return done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        cmd = tools.CommandStruct()
        T = 600
        dt = 0.02
        nIter = round(T/dt)
        cpg = {
            'initLength': 250,
            'w_y': 2.0,
            'bodyHeight':0.13,
            'bodyHeightReached':False,
            'zDist':0,
            'zHistory':np.ones((1,10)),
            'zHistoryCnt':0,
            'direction': np.ones((1,6)),
            'x':3 * np.array([[.11, -.1, .1, -.01, .12, -.12]]+[[0, 0, 0, 0, 0, 0] for i in range(30000)]),
            'y':np.zeros((30000+1,6)),
            'forward': np.ones((1,6)),
            'backward': -1 * np.ones((1,6)),
            'leftturn': [1, -1, 1, -1, 1, -1],
            'rightturn': [-1, 1, -1, 1, -1, 1],
            'legs': np.zeros((1,18)),
            'requestedLegPositions': np.zeros((3,6)),
            'correctedLegPositions': np.zeros((3,6)),
            'realLegPositions': np.zeros((3,6)),
            #'smk': smk,
            'isStance':np.zeros((1,6)),
            'pose': np.identity(3),
            'move':True,
            'groundNorm':np.zeros((1,3)),
            'groundD': 0,
            'gravVec':np.zeros((3,1)),
            'planePoint': [[0], [0], [-1.0]],
            'theta2':0,
            'theta3':0,
            'theta2Trap': 0,
            'groundTheta':np.zeros((1,30000)),
            'yOffset':np.zeros((1,6)),
            'eOffset':np.zeros((1,6)),
            'theta3Trap': 0,
            'planeTemp':np.zeros((3,3)),
            'feetTemp':np.zeros((3,6)),
            'yReq':0,
            'o':0,
            'poseLog':[],
            'feetLog':[],
            'planeLog':[]
            }

        cpg['zHistory'] = cpg['zHistory'] * cpg['bodyHeight']
        
        shoulders2 = list(range(2,18,3))
        elbows = list(range(3,18,3))

        sampleIter = 600
        cnt = 0

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
        if action == 0: #FORWARD

            
            while cnt<230:
                cpg['direction']= cpg['forward']
                cpg = CPGgs(cpg, cnt, dt)
                cpg['feetLog'].append(cpg['feetTemp'])
                cmd.position = cpg['legs']
                cnt=cnt +1
                self.pub['L'+'1'+'_'+'1'].publish(cmd.position[0][0])
                self.pub['L'+'1'+'_'+'2'].publish(cmd.position[0][1])
                self.pub['L'+'1'+'_'+'3'].publish(cmd.position[0][2])
                self.pub['L'+'6'+'_'+'1'].publish(cmd.position[0][3])
                self.pub['L'+'6'+'_'+'2'].publish(cmd.position[0][4])
                self.pub['L'+'6'+'_'+'3'].publish(cmd.position[0][5])
                self.pub['L'+'2'+'_'+'1'].publish(cmd.position[0][6])
                self.pub['L'+'2'+'_'+'2'].publish(cmd.position[0][7])
                self.pub['L'+'2'+'_'+'3'].publish(cmd.position[0][8])
                self.pub['L'+'5'+'_'+'1'].publish(cmd.position[0][9])
                self.pub['L'+'5'+'_'+'2'].publish(cmd.position[0][10])
                self.pub['L'+'5'+'_'+'3'].publish(cmd.position[0][11])
                self.pub['L'+'3'+'_'+'1'].publish(cmd.position[0][12])
                self.pub['L'+'3'+'_'+'2'].publish(cmd.position[0][13])
                self.pub['L'+'3'+'_'+'3'].publish(cmd.position[0][14])
                self.pub['L'+'4'+'_'+'1'].publish(cmd.position[0][15])
                self.pub['L'+'4'+'_'+'2'].publish(cmd.position[0][16])
                self.pub['L'+'4'+'_'+'3'].publish(cmd.position[0][17])

            
        elif action == 1: #LEFT

            while cnt < 230:
                cpg['direction']= cpg['leftturn']
                cpg = CPGgs(cpg, cnt, dt)
                cpg['feetLog'].append(cpg['feetTemp'])
                cmd.position = cpg['legs']
                cnt=cnt +1
                self.pub['L'+'1'+'_'+'1'].publish(cmd.position[0][0])
                self.pub['L'+'1'+'_'+'2'].publish(cmd.position[0][1])
                self.pub['L'+'1'+'_'+'3'].publish(cmd.position[0][2])
                self.pub['L'+'6'+'_'+'1'].publish(cmd.position[0][3])
                self.pub['L'+'6'+'_'+'2'].publish(cmd.position[0][4])
                self.pub['L'+'6'+'_'+'3'].publish(cmd.position[0][5])
                self.pub['L'+'2'+'_'+'1'].publish(cmd.position[0][6])
                self.pub['L'+'2'+'_'+'2'].publish(cmd.position[0][7])
                self.pub['L'+'2'+'_'+'3'].publish(cmd.position[0][8])
                self.pub['L'+'5'+'_'+'1'].publish(cmd.position[0][9])
                self.pub['L'+'5'+'_'+'2'].publish(cmd.position[0][10])
                self.pub['L'+'5'+'_'+'3'].publish(cmd.position[0][11])
                self.pub['L'+'3'+'_'+'1'].publish(cmd.position[0][12])
                self.pub['L'+'3'+'_'+'2'].publish(cmd.position[0][13])
                self.pub['L'+'3'+'_'+'3'].publish(cmd.position[0][14])
                self.pub['L'+'4'+'_'+'1'].publish(cmd.position[0][15])
                self.pub['L'+'4'+'_'+'2'].publish(cmd.position[0][16])
                self.pub['L'+'4'+'_'+'3'].publish(cmd.position[0][17])
            cnt=0    
            while cnt<130:
                cpg['direction']= cpg['forward']
                cpg = CPGgs(cpg, cnt, dt)
                cpg['feetLog'].append(cpg['feetTemp'])
                cmd.position = cpg['legs']
                cnt=cnt +1
                self.pub['L'+'1'+'_'+'1'].publish(cmd.position[0][0])
                self.pub['L'+'1'+'_'+'2'].publish(cmd.position[0][1])
                self.pub['L'+'1'+'_'+'3'].publish(cmd.position[0][2])
                self.pub['L'+'6'+'_'+'1'].publish(cmd.position[0][3])
                self.pub['L'+'6'+'_'+'2'].publish(cmd.position[0][4])
                self.pub['L'+'6'+'_'+'3'].publish(cmd.position[0][5])
                self.pub['L'+'2'+'_'+'1'].publish(cmd.position[0][6])
                self.pub['L'+'2'+'_'+'2'].publish(cmd.position[0][7])
                self.pub['L'+'2'+'_'+'3'].publish(cmd.position[0][8])
                self.pub['L'+'5'+'_'+'1'].publish(cmd.position[0][9])
                self.pub['L'+'5'+'_'+'2'].publish(cmd.position[0][10])
                self.pub['L'+'5'+'_'+'3'].publish(cmd.position[0][11])
                self.pub['L'+'3'+'_'+'1'].publish(cmd.position[0][12])
                self.pub['L'+'3'+'_'+'2'].publish(cmd.position[0][13])
                self.pub['L'+'3'+'_'+'3'].publish(cmd.position[0][14])
                self.pub['L'+'4'+'_'+'1'].publish(cmd.position[0][15])
                self.pub['L'+'4'+'_'+'2'].publish(cmd.position[0][16])
                self.pub['L'+'4'+'_'+'3'].publish(cmd.position[0][17])    


        
            
           

        elif action == 2: #RIGHT
            while cnt < 230:
                cpg['direction']= cpg['rightturn']
                cpg = CPGgs(cpg, cnt, dt)
                cpg['feetLog'].append(cpg['feetTemp'])
                cmd.position = cpg['legs']
                cnt=cnt +1
                self.pub['L'+'1'+'_'+'1'].publish(cmd.position[0][0])
                self.pub['L'+'1'+'_'+'2'].publish(cmd.position[0][1])
                self.pub['L'+'1'+'_'+'3'].publish(cmd.position[0][2])
                self.pub['L'+'6'+'_'+'1'].publish(cmd.position[0][3])
                self.pub['L'+'6'+'_'+'2'].publish(cmd.position[0][4])
                self.pub['L'+'6'+'_'+'3'].publish(cmd.position[0][5])
                self.pub['L'+'2'+'_'+'1'].publish(cmd.position[0][6])
                self.pub['L'+'2'+'_'+'2'].publish(cmd.position[0][7])
                self.pub['L'+'2'+'_'+'3'].publish(cmd.position[0][8])
                self.pub['L'+'5'+'_'+'1'].publish(cmd.position[0][9])
                self.pub['L'+'5'+'_'+'2'].publish(cmd.position[0][10])
                self.pub['L'+'5'+'_'+'3'].publish(cmd.position[0][11])
                self.pub['L'+'3'+'_'+'1'].publish(cmd.position[0][12])
                self.pub['L'+'3'+'_'+'2'].publish(cmd.position[0][13])
                self.pub['L'+'3'+'_'+'3'].publish(cmd.position[0][14])
                self.pub['L'+'4'+'_'+'1'].publish(cmd.position[0][15])
                self.pub['L'+'4'+'_'+'2'].publish(cmd.position[0][16])
                self.pub['L'+'4'+'_'+'3'].publish(cmd.position[0][17])

            cnt=0    
            while cnt<130:
                cpg['direction']= cpg['forward']
                cpg = CPGgs(cpg, cnt, dt)
                cpg['feetLog'].append(cpg['feetTemp'])
                cmd.position = cpg['legs']
                cnt=cnt +1
                self.pub['L'+'1'+'_'+'1'].publish(cmd.position[0][0])
                self.pub['L'+'1'+'_'+'2'].publish(cmd.position[0][1])
                self.pub['L'+'1'+'_'+'3'].publish(cmd.position[0][2])
                self.pub['L'+'6'+'_'+'1'].publish(cmd.position[0][3])
                self.pub['L'+'6'+'_'+'2'].publish(cmd.position[0][4])
                self.pub['L'+'6'+'_'+'3'].publish(cmd.position[0][5])
                self.pub['L'+'2'+'_'+'1'].publish(cmd.position[0][6])
                self.pub['L'+'2'+'_'+'2'].publish(cmd.position[0][7])
                self.pub['L'+'2'+'_'+'3'].publish(cmd.position[0][8])
                self.pub['L'+'5'+'_'+'1'].publish(cmd.position[0][9])
                self.pub['L'+'5'+'_'+'2'].publish(cmd.position[0][10])
                self.pub['L'+'5'+'_'+'3'].publish(cmd.position[0][11])
                self.pub['L'+'3'+'_'+'1'].publish(cmd.position[0][12])
                self.pub['L'+'3'+'_'+'2'].publish(cmd.position[0][13])
                self.pub['L'+'3'+'_'+'3'].publish(cmd.position[0][14])
                self.pub['L'+'4'+'_'+'1'].publish(cmd.position[0][15])
                self.pub['L'+'4'+'_'+'2'].publish(cmd.position[0][16])
                self.pub['L'+'4'+'_'+'3'].publish(cmd.position[0][17])

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
                image_data = rospy.wait_for_message('/sensors/cam/im_raw', Image, timeout=5)
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
        if not done:
            if action == 0:
                reward = 1 / float(center_detour+1)
            elif action_sum > 45: #L or R looping
                reward = -0.5
            else: #L or R no looping
                reward = 0.5 / float(center_detour+1)
        else:
            reward = -100

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

        self.last50actions = [0] * 50 #used for looping avoidance

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
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
                image_data = rospy.wait_for_message('/sensors/cam/im_raw', Image, timeout=5)
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
        return state

        # test STACK 4
        #self.s_t = np.stack((cv_image, cv_image, cv_image, cv_image), axis=0)
        #self.s_t = self.s_t.reshape(1, self.s_t.shape[0], self.s_t.shape[1], self.s_t.shape[2])
        #return self.s_t
