#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import WrenchStamped
from control_msgs.msg import JointControllerState
import tf2_ros
import tf2_geometry_msgs
import tf
import numpy as np

no_of_joints = 18
no_of_torque_directions = 6
no_legs = 6


class robot_state:
	#The following are the states that we check for a particular time!
	
        imu_state = [];
	joint_torque = np.zeros(no_of_joints*no_of_torque_directions)
	joint_positions = np.zeros(no_of_joints)
	joint_velocities = np.zeros(no_of_joints)
	robot_state = np.asarray([0, 0, 0, 0, 0, 0, 1]) #3 Trans vector and 4 quat variables!
	collision
        end_effector_z = np.zeros(no_legs) 
	
	#Call back definition for the IMU on the robot 
	def imu_callback(self, imu_data):
    	    self.imu_state = np.asarray([imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z, imu_data.orientation.w, imu_data.angular_velocity.x, imu_data.angular_velocity.y, imu_data.angular_velocity.z, imu_data.angular_velocity.y, imu_data.linear_acceleration.x, imu_data.linear_acceleration.y, imu_data.linear_acceleration.z]) 


	#Joint states callback functions	
	def joint_state_callback(self, state_data, joint_number):
                self.joint_positions[joint_number-1] = np.asarray([state_data.process_value])
                self.joint_velocities[joint_number-1] = np.asarray([state_data.process_value_dot])

	#Hard code alert! Hardcoded the array indexes. 
	#The next set of functions are the call back functions that set the joint torques. THe torque sensing has 6 outputs in 3 force directions and 3 torque directions. The state will be one array of all the states, let them be updates as convenient. Felt no need to put locks here. It shouldn't affect the calculations much 

	#The joint feedback is saved in this variable. 
	def torque_joint_callback(self, joint_torque, joint_number):
		self.joint_torque[(joint_number-1)*6:joint_number*6] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])


def listener(current_robot_state):

    rospy.init_node('state_listener', anonymous=True)


    #ALl the call back definitions!

    #Torque sensors callback!
    rospy.Subscriber("/snake_monster/sensors/imu", Imu,  current_robot_state.imu_callback)
    rospy.Subscriber("/snake_monster/sensors/SA001__MoJo/torque", WrenchStamped ,  current_robot_state.torque_joint_callback, 1)
    rospy.Subscriber("/snake_monster/sensors/SA002__MoJo/torque", WrenchStamped , current_robot_state.torque_joint_callback, 2)
    rospy.Subscriber("/snake_monster/sensors/SA003__MoJo/torque", WrenchStamped , current_robot_state.torque_joint_callback, 3)
    rospy.Subscriber("/snake_monster/sensors/SA005__MoJo/torque", WrenchStamped , current_robot_state.torque_joint_callback, 4)
    rospy.Subscriber("/snake_monster/sensors/SA0013__MoJo/torque", WrenchStamped, current_robot_state.torque_joint_callback, 5)
    rospy.Subscriber("/snake_monster/sensors/SA0014__MoJo/torque", WrenchStamped, current_robot_state.torque_joint_callback, 6)
    rospy.Subscriber("/snake_monster/sensors/SA0022__MoJo/torque", WrenchStamped, current_robot_state.torque_joint_callback, 7)
    rospy.Subscriber("/snake_monster/sensors/SA0023__MoJo/torque", WrenchStamped, current_robot_state.torque_joint_callback, 8)
    rospy.Subscriber("/snake_monster/sensors/SA0030__MoJo/torque", WrenchStamped, current_robot_state.torque_joint_callback, 9)
    rospy.Subscriber("/snake_monster/sensors/SA0031__MoJo/torque", WrenchStamped, current_robot_state.torque_joint_callback, 10)
    rospy.Subscriber("/snake_monster/sensors/SA0036__MoJo/torque", WrenchStamped, current_robot_state.torque_joint_callback, 11)
    rospy.Subscriber("/snake_monster/sensors/SA0037__MoJo/torque", WrenchStamped, current_robot_state.torque_joint_callback, 12)
    rospy.Subscriber("/snake_monster/sensors/SA0038__MoJo/torque", WrenchStamped, current_robot_state.torque_joint_callback, 13)
    rospy.Subscriber("/snake_monster/sensors/SA0043__MoJo/torque", WrenchStamped, current_robot_state.torque_joint_callback, 14)
    rospy.Subscriber("/snake_monster/sensors/SA0046__MoJo/torque", WrenchStamped, current_robot_state.torque_joint_callback, 15)
    rospy.Subscriber("/snake_monster/sensors/SA0048__MoJo/torque", WrenchStamped, current_robot_state.torque_joint_callback, 16)
    rospy.Subscriber("/snake_monster/sensors/SA0049__MoJo/torque", WrenchStamped, current_robot_state.torque_joint_callback, 17)
    rospy.Subscriber("/snake_monster/sensors/SA0050__MoJo/torque", WrenchStamped, current_robot_state.torque_joint_callback, 18)

    #State of each joint!
    rospy.Subscriber("/snake_monster/L1_1_eff_pos_controller/state", JointControllerState,  current_robot_state.joint_state_callback, (1))
    rospy.Subscriber("/snake_monster/L1_2_eff_pos_controller/state", JointControllerState,  current_robot_state.joint_state_callback, (2))
    rospy.Subscriber("/snake_monster/L1_3_eff_pos_controller/state", JointControllerState,  current_robot_state.joint_state_callback, (3))

    rospy.Subscriber("/snake_monster/L2_1_eff_pos_controller/state",  JointControllerState, current_robot_state.joint_state_callback, (4))
    rospy.Subscriber("/snake_monster/L2_2_eff_pos_controller/state",  JointControllerState, current_robot_state.joint_state_callback, (5))
    rospy.Subscriber("/snake_monster/L2_3_eff_pos_controller/state",  JointControllerState, current_robot_state.joint_state_callback, (6))

    rospy.Subscriber("/snake_monster/L3_1_eff_pos_controller/state",  JointControllerState, current_robot_state.joint_state_callback, (7))
    rospy.Subscriber("/snake_monster/L3_2_eff_pos_controller/state",  JointControllerState, current_robot_state.joint_state_callback, (8))
    rospy.Subscriber("/snake_monster/L3_3_eff_pos_controller/state",  JointControllerState, current_robot_state.joint_state_callback, (9))
    
    rospy.Subscriber("/snake_monster/L4_1_eff_pos_controller/state",  JointControllerState, current_robot_state.joint_state_callback, (10))
    rospy.Subscriber("/snake_monster/L4_2_eff_pos_controller/state",  JointControllerState, current_robot_state.joint_state_callback, (11))
    rospy.Subscriber("/snake_monster/L4_3_eff_pos_controller/state",  JointControllerState, current_robot_state.joint_state_callback, (12))
    

    rospy.Subscriber("/snake_monster/L5_1_eff_pos_controller/state",  JointControllerState, current_robot_state.joint_state_callback, (13))
    rospy.Subscriber("/snake_monster/L5_2_eff_pos_controller/state",  JointControllerState, current_robot_state.joint_state_callback, (14))
    rospy.Subscriber("/snake_monster/L5_3_eff_pos_controller/state",  JointControllerState, current_robot_state.joint_state_callback, (15))
    
    rospy.Subscriber("/snake_monster/L6_1_eff_pos_controller/state",   JointControllerState,current_robot_state.joint_state_callback, (16))
    rospy.Subscriber("/snake_monster/L6_2_eff_pos_controller/state",   JointControllerState,current_robot_state.joint_state_callback, (17))
    rospy.Subscriber("/snake_monster/L6_3_eff_pos_controller/state",   JointControllerState,current_robot_state.joint_state_callback, (18))


    #Code to get the TF for the end effector!
    listener = tf.TransformListener()
    rate = rospy.Rate(100.0)
    while not rospy.is_shutdown():
        try:

		(trans,rot)  = listener.lookupTransform('elbow__leg6__AFTER_CORNER_BODY', 'foot__leg4__INPUT_INTERFACE', rospy.Time(0))
	except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        	continue

        rate.sleep()
    
    rospy.spin()

if __name__ == '__main__':
    current_robot_state = robot_state()
    listener(current_robot_state)
