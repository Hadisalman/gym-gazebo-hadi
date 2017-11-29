import gym
import rospy
import roslaunch
import time
import numpy as np
import math
import pdb

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

from reward_functions import reward_function
import time

import moveit_commander
#from moveit_msgs.msg import RobotState
#from sensor_msgs.msg import JointState
from snake_monster.srv import *




no_of_joints = 18
no_of_torque_directions = 6
no_legs = 6
leg_contact_threshold = 0.027
state_dim = 185
action_dimension = 18
lol = 0
model = 'cpg' #define model here
#Self collision detection function

# Define Global Variables for logging:
state_sample_cnt = 0
state_sample_done = False
log_imu_state = np.zeros([100,9])
log_joint_torque = np.zeros([100,108])
log_robot_pose = np.zeros([100, 6])


def is_self_collision():
    rospy.wait_for_service('check_hit', 2)
    try:
        collision_check = rospy.ServiceProxy('check_hit', check_hit)
        resp1 = collision_check()
        print resp1.status
        return resp1.status
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e



class robot_state:
    #The following are the states that we check for a particular time!
    imu_state = [];
    joint_torque = np.zeros(no_of_joints*no_of_torque_directions)
    joint_positions = np.zeros(no_of_joints)
    joint_velocities = np.zeros(no_of_joints)
    robot_pose = np.asarray([0, 0, 0, 0, 0, 0, 1]) #3 Trans vector and 4 quat variables!
    end_effector_z = np.zeros(no_legs) 
    end_effector_angles = np.zeros(no_legs*3) 
    
    #Call back definition for the IMU on the robot 10 vals 
    def imu(self, imu_data):
        self.imu_state = np.asarray([imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z, imu_data.orientation.w, imu_data.angular_velocity.x, imu_data.angular_velocity.y, imu_data.angular_velocity.z, imu_data.linear_acceleration.x, imu_data.linear_acceleration.y, imu_data.linear_acceleration.z]) 

    #Joint states callback functions    18 + 18 = 36 vals
    def joint_state(self, state_data, joint_number):
        self.joint_positions[joint_number-1] = np.asarray([state_data.process_value])
        self.joint_velocities[joint_number-1] = np.asarray([state_data.process_value_dot])

    #Hard code alert! Hardcoded the array indexes. 
    #The next set of functions are the call back functions that set the joint torques. THe torque sensing has 6 outputs in 3 force directions and 3 torque directions. The state will be one array of all the states, let them be updates as convenient. Felt no need to put locks here. It shouldn't affect the calculations much 

    #The joint feedback is saved in this variable. 6*18  = 108
    def torque_joint(self, joint_torque, joint_number):
        self.joint_torque[(joint_number-1)*6:joint_number*6] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])

    def serialized_state(self):
        torque = []
        total_serialized = []
        #for I in range(no_of_joints):
        temp = self.joint_torque
        temp = temp.tolist()
        torque.extend(temp)
        total_serialized.extend(torque)
        total_serialized.extend(self.imu_state.tolist())
        total_serialized.extend(self.joint_positions.tolist())
        total_serialized.extend(self.joint_velocities.tolist())
	pose_list =  self.robot_pose.tolist()
	total_serialized.extend(pose_list[0])
	total_serialized.extend(pose_list[1])
	total_serialized.extend(self.end_effector_z.tolist())
	total_serialized.extend(self.end_effector_angles.tolist())
        return total_serialized

class GazeboCustomSnakeMonsterDDPG(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCustomSnakeMonsterDDPG_v0.launch")
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        #self.action_space = spaces.Discrete(3) #F,L,R
        self.reward_range = (-np.inf, np.inf)
        
	#Change Here!!!
	pi = 3.142
	lower_bound_actions = np.array([-1.22E+00,6.28E-01,-1.38E+00,8.73E-01,6.28E-01,-1.37E+00,-1.75E-01,6.28E-01,-1.43E+00,-1.75E-01,6.28E-01,-1.43E+00,8.72E-01,6.28E-01,-1.38E+00,-1.22E+00,6.28E-01,-1.38E+00])
	upper_bound_actions = np.array([-8.73E-01,1.24E+00,-8.18E-01,1.22E+00,1.22E+00,-8.19E-01,1.75E-01,1.24E+00,-1.11E+00,1.75E-01,1.24E+00,-1.11E+00,1.22E+00,1.24E+00,-8.18E-01,-8.72E-01,1.24E+00,-8.18E-01])
	lower_bound_actions = np.reshape(lower_bound_actions,(1,action_dimension))
	upper_bound_actions = np.reshape(upper_bound_actions,(1,action_dimension))
	self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(1,action_dimension))
        self.observation_space = spaces.Box(low=-2*pi, high=2*pi, shape=(1,state_dim))
	
	#time.sleep(2)
	self.previous_state = robot_state()
	self.current_state = robot_state()
	self.start = 1

        # rospy.init_node('walking_controller', anonymous=True)
        self.pub={}
        
        for i in xrange(6) :
            for j in xrange(3) :
                self.pub['L' + str(i+1) + '_' + str(j+1)] = rospy.Publisher( '/snake_monster' + '/' + 'L' + str(i+1) + '_' + str(j+1) + '_'
                                            + 'eff_pos_controller' + '/command',
                                            Float64, queue_size=10 )

	#publishers for reward function
	self.reward_pub = {}
	self.reward_pub['slip_reward'] = rospy.Publisher('/snake_monster/reward/slip_reward',Float64, queue_size=10)
	self.reward_pub['control_reward'] = rospy.Publisher('/snake_monster/reward/control_reward',Float64, queue_size=10)
	self.reward_pub['collision_reward'] = rospy.Publisher('/snake_monster/reward/collision_reward',Float64, queue_size=10)
	self.reward_pub['energy_reward'] = rospy.Publisher('/snake_monster/reward/energy_reward',Float64, queue_size=10) 
	self.reward_pub['contact_reward'] = rospy.Publisher('/snake_monster/reward/contact_reward',Float64, queue_size=10)
	self.reward_pub['movement_reward'] = rospy.Publisher('/snake_monster/reward/movement_reward',Float64, queue_size=10)
	self.reward_pub['acceleration_reward'] = rospy.Publisher('/snake_monster/reward/acceleration_reward',Float64,queue_size=10)	

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def get_state(self,reset_flag):	
	#################### State collection routine!! ##############
        #Imu data!
        current_data = robot_state()
	flag = False
	

    	listener = tf.TransformListener()
	#if reset_flag:
		#listener.clear()
    	rate = rospy.Rate(100.0)
        while flag == False:

            try:

                imu_data = rospy.wait_for_message("/snake_monster/sensors/imu", Imu, 0.1)
                current_data.imu(imu_data) 
                flag = True

            except:
                pass
        
        
        torque_ids = ['01', '02', '03', '05', '13', '14', '22', '23', '30', '31', '36', '37', '38', '43', '46', '48', '49', '50']       
        for i in range(no_of_joints):
            try:
                torque_data = rospy.wait_for_message("/snake_monster/sensors/SA0" + torque_ids[i] +  "__MoJo/torque", WrenchStamped , 0.1 )

                current_data.torque_joint(torque_data, i+1)
                
            except:
                i = i-1 #Retry to get the value
                pass

        #State of each joint!
        for limb in range(no_legs+1):
            for joint_no in range(4):
                #print limb, joint_no
                try:
                    joint_data = rospy.wait_for_message("/snake_monster/L" + str(limb+1) + "_" + str(joint_no+1) + "_eff_pos_controller/state", JointControllerState, 0.1 )
                    
                    current_data.joint_state(joint_data, limb*3 + joint_no + 1)
                    
                except:
                    joint_no = joint_no -1 #Retry to get the value
                    pass
        #State of each foot!    

        for limb in range(no_legs):
            try:
                (trans,rot)  = listener.lookupTransform('map', 'foot__leg' + str(limb+1) + '__INPUT_INTERFACE', rospy.Time(0))
                current_data.end_effector_z[limb] = (trans[2] <  leg_contact_threshold)*1
                quaternion = (
                    rot[0],
                    rot[1],
                    rot[2],
                    rot[3])
                euler = tf.transformations.euler_from_quaternion(quaternion)

                current_data.end_effector_angles[3*limb:3*limb + 3] = euler #Roll pitch yaw
                #print current_data.end_effector_angles[3*limb:3*limb + 3], limb

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                limb = limb - 1 #Check until you get the value
                pass
        
	#Robot position transform!
        flag = False
        while flag == False:
            try:
                (trans,rot)  = listener.lookupTransform('base', 'map', rospy.Time(0))
                current_data.robot_pose = np.asarray([trans, rot])
                flag = True
           
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
	done = 0
	self.previous_state = current_data;
	return current_data





    def _step(self, action):
	print "in step"	
	previous_state = self.previous_state 

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


            #if action == 0: #FORWARD
	global model
	global lol 
	log_joint_angles = np.zeros([100,18])
        sampling_done = False
	sample_count = -1
	if model == 'cpg':
	#only forward action is taken
		while cnt <230:
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
			#if cnt % 2 == 0 and sampling_done == False and sample_count<100:
			#	sample_count += 1 
			#	log_joint_angles[sample_count-1][:] = cmd.position[0][:]
			#	if sample_count == log_joint_angles.shape[0]-1 :
			#		np.savetxt('joint_angles.csv',log_joint_angles,delimiter=",")
			#		sampling_done == True
			
	elif model == 'ddpg' :
		start = int(round(time.time() * 1000))  	
 		while abs(int(round(time.time() * 1000)) - start)< 500:       	
			
	      		self.pub['L'+'1'+'_'+'1'].publish(action[0])
        		self.pub['L'+'1'+'_'+'2'].publish(action[1])
        		self.pub['L'+'1'+'_'+'3'].publish(action[2])
			self.pub['L'+'6'+'_'+'1'].publish(action[3])
			self.pub['L'+'6'+'_'+'2'].publish(action[4])
			self.pub['L'+'6'+'_'+'3'].publish(action[5])
			self.pub['L'+'2'+'_'+'1'].publish(action[6])
			self.pub['L'+'2'+'_'+'2'].publish(action[7])
			self.pub['L'+'2'+'_'+'3'].publish(action[8])
			self.pub['L'+'5'+'_'+'1'].publish(action[9])
			self.pub['L'+'5'+'_'+'2'].publish(action[10])
			self.pub['L'+'5'+'_'+'3'].publish(action[11])
			self.pub['L'+'3'+'_'+'1'].publish(action[12])
			self.pub['L'+'3'+'_'+'2'].publish(action[13])
			self.pub['L'+'3'+'_'+'3'].publish(action[14])
			self.pub['L'+'4'+'_'+'1'].publish(action[15])
			self.pub['L'+'4'+'_'+'2'].publish(action[16])
			self.pub['L'+'4'+'_'+'3'].publish(action[17])
			
			#if lol == 0 : 
			#	self.pub['L'+'4'+'_'+'3'].publish(0)
			#	lol = 1
			#else:
			#	self.pub['L'+'4'+'_'+'3'].publish(np.pi/2)
			#	lol = 0

	
        data = None
        #current_data = robot_state()
   	#Collision check
   	print is_self_collision(), "Collision check is"
   
        rospy.wait_for_service('/gazebo/pause_physics')
	
	current_state = self.get_state(False)
	
	global state_sample_done
	global sample_count
	global state_sample_cnt
	global log_imu_state
	global log_joint_torque
	global log_robot_pose

	if state_sample_done == False and sample_count<30:
		state_sample_cnt += 1
		q1 = np.array(current_state.robot_pose[1])
		euler_pose = np.array(tf.transformations.euler_from_quaternion(q1))
		q2 = current_state.imu_state[0:4]	
		euler_imu = np.array(tf.transformations.euler_from_quaternion(q2))
		log_imu_state[state_sample_cnt-1,:3] = euler_imu[:]
		log_imu_state[state_sample_cnt-1,3:] = current_state.imu_state[4:]
		#pdb.set_trace()
		log_joint_torque[state_sample_cnt-1][:] = current_state.joint_torque[:]
		log_robot_pose[state_sample_cnt-1,:3] = current_state.robot_pose[0]
		#pdb.set_trace()
		log_robot_pose[state_sample_cnt-1,3:] = euler_pose
		#pdb.set_trace()
		if(state_sample_cnt == 30):
			np.savetxt('imu_state_cpg.csv',log_imu_state,delimiter=",")
			np.savetxt('joint_torques_cpg.csv',log_joint_torque,delimiter=",")
			np.savetxt('robot_pose_cpg.csv',log_robot_pose,delimiter=",")
			state_sample_done = True
			print("CSVs saved!")

	try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")


	if(self.start != 1):
		reward_object = reward_function(previous_state, current_state)	
		reward = reward_object.total_reward()
		self.publish_reward(reward_object) # publishes the reward function individual values
	else:
		self.start = 0	
		reward = 1
		
	done = 0
	serialized_state  = current_state.serialized_state()
        #return current_state, reward, done
	print "Returning from step"
	return serialized_state, reward, done, {}

    def _reset(self):
	
	#global reset_flag
	#reset_flag = True
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

        
	rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")

	#listener = tf.TransformListener()
	#listener.clear()
	current_state = self.get_state(True) 
	#serialized_state  = current_state.serialized_state()
        
	return state_dim*[0]


    def publish_reward(self,reward_object):
	self.reward_pub['slip_reward'].publish(reward_object.slip_avoidance())
	self.reward_pub['control_reward'].publish(reward_object.control_input())
	self.reward_pub['collision_reward'].publish(reward_object.self_collision())
	self.reward_pub['energy_reward'].publish(reward_object.conservation_of_energy())
	self.reward_pub['contact_reward'].publish(reward_object.ground_contact())
	self.reward_pub['movement_reward'].publish(reward_object.any_movement())
	self.reward_pub['acceleration_reward'].publish(reward_object.forward_acceleration())

    def term_cond(self,state,time):
	roll_thres = float(3.1415/2) # Hard-coded using observed values (CURRENTLY ARBITRARY)
	pitch_thres = float(3.1415/3) # Hard-coded using observed values (CURRENTLY ARBITRARY)
	terminal_time = 1800 # Milliseconds
	premat = 1 # Set premature end -> true as default
	done = 0 # Set done -> incomplete as default
	if((abs(state.imu_state[0])>roll_thres)or(abs(state.imu_state[1])>=pitch_thres)or(state.joint_positions>=joint_angles_lim)): # Robot orientation||Joint angles indicate collision, end episode
	    done = 1
	    premat = 1
	elif(time >= terminal_time): # Time requirement met, end episode
	    premat = 0
	    done = 1
	else:
	    # Terminal conditions not met, do not end episode
	    done = 0 
	    premat = None
	return done, premat	


