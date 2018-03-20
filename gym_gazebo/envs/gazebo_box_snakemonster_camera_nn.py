import gym
import rospy
import roslaunch
import time
import numpy as np
import cv2
import sys
import os
import random
import getModelStates
import math

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

import tools
from Functions.Controller import Controller
from Functions.CPGgs import CPGgs
from std_srvs.srv import Empty
from std_msgs.msg import Float64
from gazebo_msgs.srv import GetModelState

from IPython import embed

class GazeboBoxSnakeMonsterCameraNnEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboBoxSnakeMonsterCamera_v0.launch")
        # self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.state_pub = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=10)
        self.modelCoordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.action_space = spaces.Discrete(3) #F,L,R
        self.reward_range = (-np.inf, np.inf)

        self.pub={}
        self.rate=rospy.Rate(100)
        for i in xrange(6) :
            for j in xrange(3) :
                self.pub['L' + str(i+1) + '_' + str(j+1)] = rospy.Publisher( '/snake_monster' + '/' + 'L' + str(i+1) + '_' + str(j+1) + '_'
                                            + 'eff_pos_controller' + '/command',
                                            Float64, queue_size=100 )

        self._seed()
        self.episode = 0
        self.last50actions = [0] * 50
        self.last5rewards = [0] * 5
        self.img_rows = 84
        self.img_cols = 84
        self.img_channels = 1
        self.endXPosition = -6.5
        self.steps = 0

    def setmodelstate(self, modelname='robot',x=0,y=0,yaw=0):
        # rospy.init_node('ali')    
        state=ModelState()
        state.model_name = modelname
        state.pose.position.x=x
        state.pose.position.y=y
        state.pose.orientation.z=yaw
        self.state_pub.publish(state)

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

    def quaternion_to_euler_angle(self, w, x, y, z):
        ysqr = y * y
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = (math.atan2(t0, t1))
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = (math.asin(t2))
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = (math.atan2(t3, t4))
        
        return X, Y, Z

    def _step(self, action):
        self.steps +=1
        startState = self.modelCoordinates('robot', 'world')

        # print(startState.pose.position.x)
        cmd = tools.CommandStruct()
        T = 600
        dt = 0.02
        nIter = int(round(T/dt))

        # cpg['zHistory'] = cpg['zHistory'] * cpg['bodyHeight']
        
        forward = np.ones((1,6))
        backward = -1 * np.ones((1,6))
        leftturn = np.array([[1, -1, 1, -1, 1, -1]])
        rightturn = np.array([[-1, 1, -1, 1, -1, 1]])

        shoulders1          = list(range(0,18,3)) # joint IDs of the shoulders
        shoulders2          = list(range(1,18,3)) # joint IDs of the second shoulder joints
        elbows              = list(range(2,18,3)) # joint IDs of the elbow joints

        # cpg = {
        #     'initLength': 250, # was 250, -1 needed for random initial poses given by cpgConfig code          
        # # 'w',            60,... %TUNEABLE: CPG speed
        # # 'bodyHeight',   0.18,... %TUNEABLE: Selected body height in meters
        # # 'r',            0.10,... %TUNEABLE: Distance of center legs from body in meters
        # # 'direction',    forward,...% TUNEABLE: set to forward, backward, right (point turn), left (point turn)
        # # 'a',            45 * ones(1,6),...% TUNEABLE: step height in degrees
        # # 'b',            3.75 * ones(1,6),...% TUNEABLE: 1/2 step sweep in degrees (sweep = 2b)
        # # 's1Off',        pi/3,...% TUNEABLE: offset on proximal joint for front and back legs (determines leg spread; bigger = futher forward/backward)
        # # 's2Off',        pi/16,...% TUNEABLE: offset on intermediate joint for all legs  
        # # 't3Str',        0.0,...%% TUNEABLE: determines how much the distal moves up and out when a leg is off the ground (useful for stepping onto large obstacles)
        # # 'stabilize',    true,... % TUNEABLE: true/false: turns stabilization on or off
        # # 'x',            5 * [1 -1 1 -1 1 -1; zeros(nIter,6)],... %TUNEABLE: Initial CPG x-positions
        # # 'y',            20 * [1 1 1 1 1 1; zeros(nIter,6)],... %TUNEABLE: Initial CPG y-positions
        # # 'legs',         zeros(1,18),... %Joint angle values
        # # 'move',         true,... %true: walks according to cpg.direction, false: stands in place (will continue to stabilize); leave to true for CPG convergence
        # # 'smk',          smk,... %Snake Monster Kinematics object
        # # 'pose',         eye(3),... %SO(3) rotation matrix describing body orientation
        # # 'zHistory',     ones(1,10),... %Height correction
        # # 'zHistoryCnt',  1,... %Height correction
        # # 'theta2',       0 ... %Offset on shoulder2 (intermediate joint)
        # }

        cpg = {
            'initLength': 250, # was 250, -1 needed for random initial poses given by cpgConfig code
            'bodyHeight': 0.3,
            'r': 0.10,
            'direction': forward,
            ## SUPERELLIPSE
            # 'w': 70,
            # 'a': 45 * np.ones((1,6)),
            # 'b': 3.75 * np.ones((1,6)),
            # 'x': 5.0 * np.array([[1, -1, 1, -1, 1, -1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)]),
            # 'y': 20.0 * np.array([[1, 1, 1, 1, 1, 1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)]),
            ## /SUPERELLIPSE
            'w': 100,
            'a': 50 * np.ones((1,6)),
            'b': 3.5 * np.ones((1,6)),
            'x': 3.75 * np.array([[1, -1, 1, -1, 1, -1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)]),
            'y': 0.0 * np.array([[1, 1, 1, 1, 1, 1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)]),
            
            'b0': np.zeros((1,6)),    #Original step length
            'mu': np.array([0.0412, 0.0412, 0.0882, 0.0882, 0.0412, 0.0412]),
            's1Off': np.pi/3,
            's2Off': np.pi/16,
            't3Str': 0,
            'stabilize': True,
            #'isStance': np.zeros((1,6)),
            'dx': np.zeros((3,6)),
            'legs': np.zeros((1,18)),
            'legsAlpha': 0.75,
            'move': True,
            # 'smk': smk,
            'pose': np.identity(3),
            'zErr': 0.0,
            'zHistory': np.ones((1,10)),
            'zHistoryCnt': 1,
            'dTheta2': np.array([0.,0.,0.,0.,0.,0.]),
            'theta2': np.array([0.,0.,0.,0.,0.,0.]),
            'theta_step2': np.array([0.,0.,0.,0.,0.,0.]),
            'groundD': 0,
            'groundNorm': 0,
            'torques': np.zeros((3,6)),
            'jacobianTorques': np.zeros((3,6)),
            'phi':0,     #Angle of travel based on joystick
        }

        
        cycle=300

        cnt = 0
        # embed()

        shoulders2 = list(range(2,18,3))
        elbows = list(range(3,18,3))
        
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException, e:
            print ("/gazebo/unpause_physics service call failed")

        _, pitch, yaw = self.quaternion_to_euler_angle(startState.pose.orientation.w, startState.pose.orientation.x, startState.pose.orientation.y, startState.pose.orientation.z)
        
        cpg['w'] = 300
        cpg['a']= 50 * np.ones((1,6))
        cpg['b']= 3.5 * np.ones((1,6))
        cpg['x'] = 3.75 * np.array([[1, -1, 1, -1, 1, -1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)])
        cpg['y'] = 0.0 * np.array([[1, 1, 1, 1, 1, 1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)])
        cpg['s1Off'] = np.pi/3
        cpg['t3Str'] = 0.0

        cpg['zHistory'] = cpg['zHistory'] * cpg['bodyHeight']
        cpg['theta_min'] = -np.pi/2 - cpg['s2Off']
        cpg['theta_max'] = np.pi/2 - cpg['s2Off']

        print(pitch, yaw)
        # if(abs(pitch) <0.01):
        while(abs(yaw - 1.57) > 0.1): 
            if (yaw > 1.5708):
                cpg['direction']= rightturn
            else:
                cpg['direction']= leftturn
            cnt = 0
            while (cnt < 60):
                cpg = CPGgs(cpg, cnt, dt)
                # cpg['feetLog'].append(cpg['feetTemp'])
                cmd.position = cpg['legs']
                # embed()
                cnt += 1
                # if cnt >30:
                    # currentState = self.modelCoordinates('robot', 'world')
                    # print(currentState.pose.position.x)
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
                self.rate.sleep()
            startState = self.modelCoordinates('robot', 'world')
            _, _, yaw = self.quaternion_to_euler_angle(startState.pose.orientation.w, startState.pose.orientation.x, startState.pose.orientation.y, startState.pose.orientation.z)
            # print(yaw)

        # low steps for empty environment
        if (action == 0):
            cpg['w'] = 320
            cpg['a']= 50 * np.ones((1,6))
            cpg['b']= 3.5 * np.ones((1,6))
            cpg['x'] = 3.75 * np.array([[1, -1, 1, -1, 1, -1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)])
            cpg['y'] = 0.0 * np.array([[1, 1, 1, 1, 1, 1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)])
            cpg['s1Off'] = np.pi/3
            cpg['t3Str'] = 0.0
            # cycle = 150

        # high steps for obstacles (0.1m high plate)
        elif (action == 1):
            cpg['w'] = 140
            cpg['a'] = 30 * np.ones((1,6))
            cpg['b'] = 5 * np.ones((1,6))
            cpg['x'] = 5.0 * np.array([[1, -1, 1, -1, 1, -1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)])
            cpg['y'] = 20.0 * np.array([[1, 1, 1, 1, 1, 1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)])
            cpg['s1Off'] = np.pi/3
            cpg['t3Str'] = 1.2

        cpg['zHistory'] = cpg['zHistory'] * cpg['bodyHeight']
        cpg['theta_min'] = -np.pi/2 - cpg['s2Off']
        cpg['theta_max'] = np.pi/2 - cpg['s2Off']

        cnt = 0
        while cnt<cycle:

            cpg['direction']= forward
            cpg = CPGgs(cpg, cnt, dt)
            # cpg['feetLog'].append(cpg['feetTemp'])
            cmd.position = cpg['legs']
            # embed()
            cnt=cnt +1
            if cnt >30:
                # currentState = self.modelCoordinates('robot', 'world')
                # print(currentState.pose.position.x)
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
                self.rate.sleep()

        image_data = None
        success=False
        cv_image = None
        # while image_data is None or success is False:
        #     try:
        image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=None)
        h = image_data.height
        w = image_data.width
        cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
        #         #temporal fix, check image is not corrupted
        #         if not (cv_image[h/2,w/2,0]==178 and cv_image[h/2,w/2,1]==178 and cv_image[h/2,w/2,2]==178):
        #             success = True
        #         else:
        #             print("/camera/rgb/image_raw ERROR, retrying")
        #             pass
        #     except:
        #         pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")


        newState = self.modelCoordinates('robot', 'world')

        distancetravelled = (startState.pose.position.x - max(newState.pose.position.x, self.endXPosition))
        
        self.last5rewards.pop(0) #remove oldest
        if distancetravelled < 0.1:
            reward = -10
            self.last5rewards.append(1)
        else:
            reward = 10.0*distancetravelled - 5
            self.last5rewards.append(0)

        print("reward is: {}".format(reward))
        reward_sum = sum(self.last5rewards)

        done = False
        if(newState.pose.position.x <= self.endXPosition):
            done = True
        
        if not done:
            if reward_sum > 3:
                reward = -100
                done = True    

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


        self.steps = 0  
        self.last50actions = [0] * 50 #used for looping avoidance
        self.last5rewards = [0] * 5 #used for avoiding getting stuck

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
            # self.setmodelstate(x=0,y=0,yaw=3.14/2+3.14*(self.episode%2))

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
                image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=None)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                #temporal fix, check image is not corrupted
                if not (cv_image[h/2,w/2,0]==178 and cv_image[h/2,w/2,1]==178 and cv_image[h/2,w/2,2]==178):
                    success = True
                else:
                    print("/camera/rgb/image_raw ERROR, retrying")
                    pass
                    # print("/camera/rgb/image_raw ERROR, retrying")
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")

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
