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
from cpg import *
import numpy as np

class JointCmds:
    """
    The class provides a dictionary mapping joints to command values.
    """
    def __init__( self ) :
        
        self.joints_list = []
        for i in xrange(6) :
            for j in xrange(3) :
                leg_str='L' + str(i+1) + '_' + str(j+1)
                self.joints_list += [leg_str]

        self.cpg_joints = ['L6_2', 'L1_2', 'L4_2', 'L3_2']
        self.group_joints = [['L1_1', 'L4_1'], ['L6_1', 'L3_1'],
                             ['L2_2', 'L5_2'], ['L2_3', 'L5_3']]
        self.group_joints_flat = [item for sublist in self.group_joints \
                                  for item in sublist]
        self.cpg = CPG()
        self.jnt_cmd_dict = {}

    def update( self, dt ) :
        s = self.cpg.simulate(dt)
        x = s[:4]
        y = s[4:]

        for i, jnt in enumerate(self.cpg_joints) :
            self.jnt_cmd_dict[jnt] = max(0.5*y[i],0)

        self.jnt_cmd_dict['L1_1'] = 0.2
        self.jnt_cmd_dict['L6_1'] = -0.2
        self.jnt_cmd_dict['L4_1'] = -.75
        self.jnt_cmd_dict['L3_1'] = .75

        self.jnt_cmd_dict['L6_1'] += \
            .45*(np.cos(np.arctan2(y[0],x[0])+np.pi)+1)
        self.jnt_cmd_dict['L1_1'] -= \
            .45*(np.cos(np.arctan2(y[1],x[1])+np.pi)+1)
        self.jnt_cmd_dict['L4_1'] += \
            .45*(np.cos(np.arctan2(y[2],x[2])+np.pi)+1)
        self.jnt_cmd_dict['L3_1'] -= \
            .45*(np.cos(np.arctan2(y[3],x[3])+np.pi)+1)

        for jnt in self.group_joints[2] :
            self.jnt_cmd_dict[jnt] = 1.0
        for jnt in self.group_joints[3] :
            self.jnt_cmd_dict[jnt] = -1.0
        
        for jnt in self.joints_list :
            if jnt not in self.cpg_joints and \
               jnt not in self.group_joints_flat :
                self.jnt_cmd_dict[jnt] = 0.0
                
        return self.jnt_cmd_dict


def publish_commands( hz, pub):
    
    # rospy.init_node('walking_controller', anonymous=True)
    rate = rospy.Rate(hz)
    jntcmds = JointCmds()
    count=0
    while count<500:
        count=count + 1
        jnt_cmd_dict = jntcmds.update(1./hz)
        for jnt in jnt_cmd_dict.keys() :
           pub[jnt].publish( jnt_cmd_dict[jnt] )
        rate.sleep()


# if __name__ == "__main__":
#     try:
#         hz = 100
#         publish_commands( hz )
#         # cpg = CPG()
#         # dt=1./hz
#         # cpg.plot(10,dt)
#     except rospy.ROSInterruptException:
#         pass


flag=False
class GazeboCircuit2SnakeMonsterLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2SnakeMonsterLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/snake_monster/L1_1_eff_pos_controller/commandss', Float64, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.action_space = spaces.Discrete(3) #F,L,R
        self.reward_range = (-np.inf, np.inf)

        self._seed()

        self.pub={}
        ns_str = '/snake_monster'
        cont_str = 'eff_pos_controller'
        for i in xrange(6) :
            for j in xrange(3) :
                leg_str='L' + str(i+1) + '_' + str(j+1)
                self.pub[leg_str] = rospy.Publisher( ns_str + '/' + leg_str + '_'
                                            + cont_str + '/command',
                                            Float64, queue_size=10 )

    def discretize_observation(self,data,new_ranges):
        discretized_ranges = []
        min_range = 0.35
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

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException, e:
            print ("/gazebo/unpause_physics service call failed")

        if action == 0: #FORWARD
           
            vel_cmd = 3.14/2 
            publish_commands(100,self.pub)
            # self.vel_pub.publish(vel_cmd)
        elif action == 1: #LEFT
        
            vel_cmd=3.14/2
            publish_commands(100,self.pub)
            # self.vel_pub.publish(vel_cmd)

        elif action == 2: #RIGHT
            vel_cmd = 3.14/2
            # self.vel_pub.publish(vel_cmd)
            publish_commands(100,self.pub)
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
