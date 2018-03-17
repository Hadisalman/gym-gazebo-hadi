import gym
import rospy
import roslaunch
import time
import numpy as np

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from gym.utils import seeding
import roslib; roslib.load_manifest('gazebo_ros')
from gazebo_msgs.msg import ModelState
import getModelStates
from tf.transformations import quaternion_from_euler

class GazeboMax1TurtlebotLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboMax1TurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.action_space = spaces.Discrete(3) #F,L,R
        self.state_pub = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)

        self.reward_range = (-np.inf, np.inf)
        self.observation_space = spaces.Box(-0, 7,(20,))
        self._seed()
        self.initial_angles = [0, np.pi/2, np.pi, 3.0*np.pi/4]

    def calculate_observation(self,data):
        min_range = 0.2
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        return data.ranges, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def setmodelstate(self, modelname='mobile_base',x=0,y=0,yaw=0):
        # rospy.init_node('ali')    
        state=ModelState()
        state.model_name = modelname
        state.pose.position.x=x
        state.pose.position.y=y
        
        # yaw = np.pi/2
        q = quaternion_from_euler(0,0,yaw)

        state.pose.orientation.x = q[0]
        state.pose.orientation.y = q[1]
        state.pose.orientation.z = q[2]
        state.pose.orientation.w = q[3]

        self.state_pub.publish(state)


    def _step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException, e:
            print ("/gazebo/unpause_physics service call failed")

        # max_ang_speed = 0.3
        # ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        # vel_cmd = Twist()
        # vel_cmd.linear.x = 0.2
        # vel_cmd.angular.z = ang_vel
        # self.vel_pub.publish(vel_cmd)

        if action == 0: #FORWARD
            vel_cmd = Twist()
            
            vel_cmd.linear.x = 0.3
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1: #LEFT
            vel_cmd = Twist()
            
            vel_cmd.linear.x = 0.1
            vel_cmd.angular.z = 0.3
           
            self.vel_pub.publish(vel_cmd)
        elif action == 2: #RIGHT
            vel_cmd = Twist()
           
            vel_cmd.linear.x = 0.1
            vel_cmd.angular.z = -0.3
            
            self.vel_pub.publish(vel_cmd)

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

        # if not done:
        #     # Straight reward = 5, Max angle reward = 0.5
        #     reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
        #     # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))
        # else:
        #     reward = -200
        if not done:
            if action == 0:
                reward = 5
            else:
                reward = 0
        else:
            reward = -200

        # embed()
        return np.asarray(state), reward, done, {}

    def _reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
            self.setmodelstate(x=0,y=0,yaw=self.initial_angles[np.random.choice(4)])

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
