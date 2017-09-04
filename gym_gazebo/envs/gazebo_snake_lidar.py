import gym
import rospy
import roslaunch
import time
import numpy as np

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan

from gym.utils import seeding
from copy import copy
import time
import tools
from Functions.Controller import Controller
from Functions.CPGgs import CPGgs
import getModelStates
import geometry_msgs.msg
import sys 

class JointCmds:
    """
    The class provides a dictionary mapping joints to command values.
    """
    def __init__( self, num_mods ) :
        
        self.num_modules = num_mods
        self.jnt_cmd_dict = {}        
        self.joints_list = []
        self.t = 0.0
        
        for i in range(self.num_modules) :
            leg_str='S_'
            if i < 10 :
                leg_str += '0' + str(i)
            else :
                leg_str += str(i)
            self.joints_list += [leg_str]

    def update( self, dt ) :

        self.t += dt

        ## sidewinding gait ##
        # spatial frequency
        spat_freq = 0.6
        
        # temporal phase offset between horizontal and vertical waves
        TPO = 2.0/8.0

        # amplitude
        A = 2*np.pi/2.0

        # direction
        d = -1

        # if even
            # command = A*sin( 2.0*np.pi*(d*t + module_index*spat_freq) )
        # if odd
            # command = A*sin( 2.0*np.pi*(d*t + TPO + module_index*spat_freq) )

        for i, jnt in enumerate(self.joints_list) :
            self.jnt_cmd_dict[jnt] = A*np.sin( 2.0*np.pi*(d*self.t + (i%2)*TPO + i*spat_freq) )
                
        return self.jnt_cmd_dict


class GazeboSnakeLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboSnakeLidar_v0.launch")
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.action_space = spaces.Discrete(3) #F,L,R
        self.reward_range = (-np.inf, np.inf)
        # rospy.init_node('walking_controller', anonymous=True)
        self.pub={}
        
        for i in xrange(6) :
            for j in xrange(3) :
                self.pub['L' + str(i+1) + '_' + str(j+1)] = rospy.Publisher( '/snake_monster' + '/' + 'L' + str(i+1) + '_' + str(j+1) + '_'
                                            + 'eff_pos_controller' + '/command',
                                            Float64, queue_size=10 )


        self._seed()

    def discretize_observation(self,data,new_ranges):
        discretized_ranges = []
        min_range = 0.0004
        done = False
        mod = len(data.ranges)/new_ranges
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf') or np.isinf(data.ranges[i]):
                    discretized_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(int(data.ranges[i]))
            if (min_range > data.ranges[i] > 0):
                done = True
        return discretized_ranges,done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        pub={}
        ns_str = '/snake'
        cont_str = 'eff_pos_controller'
        for i in range(16) :
            leg_str='S_'
            if i < 10 :
                leg_str += '0' + str(i)
            else :
                leg_str += str(i)
            pub[leg_str] = rospy.Publisher( ns_str + '/' + leg_str + '_'
                                        + cont_str + '/command',
                                        Float64, queue_size=10 )
        rate = rospy.Rate(100)
        jntcmds = JointCmds(num_mods=16)
        # while not rospy.is_shutdown():
        #     jnt_cmd_dict = jntcmds.update(1./100)
        #     for jnt in jnt_cmd_dict.keys() :
        #         pub[jnt].publish( jnt_cmd_dict[jnt] )
        #     rate.sleep()
        
        cnt = 0
    
        angle = getModelStates.gms_client('ground_plane','SA008__MoJo__OUTPUT_BODY')
        print(angle.pose.position.x)
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException, e:
            print ("/gazebo/unpause_physics service call failed")

        if action == 0: #FORWARD

            while cnt<300:
                jnt_cmd_dict = jntcmds.update(1./100)
                for jnt in jnt_cmd_dict.keys() :
                    pub[jnt].publish( jnt_cmd_dict[jnt] )
                rate.sleep()

                cnt=cnt +1
            

        elif action == 1: #LEFT

            while cnt<300:
                jnt_cmd_dict = jntcmds.update(1./100)
                for jnt in jnt_cmd_dict.keys() :
                    pub[jnt].publish( jnt_cmd_dict[jnt] )
                rate.sleep()

                cnt=cnt +1
            
        

        elif action == 2: #RIGHT
           
            while cnt<300:
                jnt_cmd_dict = jntcmds.update(1./100)
                for jnt in jnt_cmd_dict.keys() :
                    pub[jnt].publish( jnt_cmd_dict[jnt] )
                rate.sleep()

                cnt=cnt +1   
           
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

        state,done = self.discretize_observation(data,5)

        if not done:
            if action == 0:
                reward = 5
            else:
                reward = 1
        else:
            reward = -200

        return state, reward, done, {}

    def _reset(self):

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

        state = self.discretize_observation(data,5) 

        return state
