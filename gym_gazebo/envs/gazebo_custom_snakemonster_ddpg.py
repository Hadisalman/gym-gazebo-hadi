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
    
import numpy as np
from copy import copy
import time
import tools
from Functions.Controller import Controller
from Functions.CPGgs import CPGgs


from sensor_msgs.msg import Imu
from geometry_msgs.msg import WrenchStamped
from control_msgs.msg import JointControllerState
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs
import tf


no_of_joints = 18
no_of_torque_directions = 6
no_legs = 6
leg_contact_threshold = 0.027

class robot_state:
	#The following are the states that we check for a particular time!
        imu_state = [];
	joint_torque = np.zeros(no_of_joints*no_of_torque_directions)
	joint_positions = np.zeros(no_of_joints)
	joint_velocities = np.zeros(no_of_joints)
	robot_pose = np.asarray([0, 0, 0, 0, 0, 0, 1]) #3 Trans vector and 4 quat variables!
        end_effector_z = np.zeros(no_legs) 
        	
	#Call back definition for the IMU on the robot 
	def imu(self, imu_data):
    	    self.imu_state = np.asarray([imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z, imu_data.orientation.w, imu_data.angular_velocity.x, imu_data.angular_velocity.y, imu_data.angular_velocity.z, imu_data.angular_velocity.y, imu_data.linear_acceleration.x, imu_data.linear_acceleration.y, imu_data.linear_acceleration.z]) 

	#Joint states callback functions	
	def joint_state(self, state_data, joint_number):
                self.joint_positions[joint_number-1] = np.asarray([state_data.process_value])
                self.joint_velocities[joint_number-1] = np.asarray([state_data.process_value_dot])

	#Hard code alert! Hardcoded the array indexes. 
	#The next set of functions are the call back functions that set the joint torques. THe torque sensing has 6 outputs in 3 force directions and 3 torque directions. The state will be one array of all the states, let them be updates as convenient. Felt no need to put locks here. It shouldn't affect the calculations much 

	#The joint feedback is saved in this variable. 
	def torque_joint(self, joint_torque, joint_number):
		self.joint_torque[(joint_number-1)*6:joint_number*6] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])





class GazeboCircuit2SnakeMonsterLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2SnakeMonsterLidar_v0.launch")
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.action_space = spaces.Discrete(3) #F,L,R
        self.reward_range = (-np.inf, np.inf)
	#self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        #self.observation_space = spaces.Box(low=-high, high=high)



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
        min_range = 0.4
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
	current_data = robot_state()
        while data is None:
            #Lidar data. We need to remove this or spoof this!
	    try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            
	    except:
                pass
            

	    #Imu data!
	    try:
    		imu_data = rospy.wait_for_message("/snake_monster/sensors/imu", Imu,5  )
		current_data.imu(imu_data) 
	    except:
		pass
    		
            try:
		torque_ids = [1, 2, 3, 5, 13, 14, 22, 23, 30, 31, 36, 37, 38, 43, 46, 48, 49, 50]		
		for i in range(no_of_joints):

			torque_data = rospy.wait_for_message("/snake_monster/sensors/SA00" + str(torque_ids[i]) +  "__MoJo/torque", WrenchStamped , 5)
			current_data.torque_joint_state(torque_data, i+1)	
	    except:
                pass

    #State of each joint!
	    for limb in range(no_legs):
		for joint_no in range(3)
    			try:
				joint_data = rospy.wait_for_message("/snake_monster/L" + str(limb) + "_" + str(joint_no) + "_eff_pos_controller/state", JointControllerState, 5)
					current_robot_state.joint_state(joint_data, limb*joint_no + 1)
			except:
				pass
	#State of each foot!	

	    for limb in range(no_legs):
            	try:
			(trans,rot)  = listener.lookupTransform('map', 'foot__leg' + str(limb) + '__INPUT_INTERFACE', rospy.Time(0))
			current_robot_state.end_effector_z[limb] = (trans[2] <  leg_contact_threshold)*1
	    	except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        		pass
	
	#Robot position transform!
	    try:
	    	(trans,rot)  = listener.lookupTransform('base', 'map', rospy.Time(0))
		current_robot_state.robot_pose = np.asarray([rot, trans])
	    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
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





