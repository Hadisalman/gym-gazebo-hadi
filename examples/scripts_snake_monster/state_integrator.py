#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import WrenchStamped

import numpy as np

no_of_joints = 18
no_of_torque_directions = 6

class robot_state:
	imu_state = [];
	joint_torque = np.zeros(no_of_joints*no_of_torque_directions)
	joint_positions
	joint_velocities
	
	def imu_callback(self, imu_data):
    		self.imu_state = np.asarray([imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z, imu_data.orientation.w, imu_data.angular_velocity.x, imu_data.angular_velocity.y, imu_data.angular_velocity.z, imu_data.angular_velocity.y, imu_data.linear_acceleration.x, imu_data.linear_acceleration.y, imu_data.linear_acceleration.z]) 
	
	#Hard code alert! Hardcoded the array indexes. 
	#The next set of functions are the call back functions that set the joint torques. THe torque sensing has 6 outputs in 3 force directions and 3 torque directions. The state will be one array of all the states, let them be updates as convenient. Felt no need to put locks here. It shouldn't affect the calculations much 

	def joint_1_callback(self, joint_torque):
		self.joint_torque[0:6] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])

	def joint_2_callback(self, joint_torque):
		self.joint_torque[6:12] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])

	def joint_3_callback(self, joint_torque):
		self.joint_torque[12:18] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])

	def joint_4_callback(self, joint_torque):
		self.joint_torque[18:24] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])

	def joint_5_callback(self, joint_torque):
		self.joint_torque[24:30] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])

	def joint_6_callback(self, joint_torque):
		self.joint_torque[30:35] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])


	def joint_7_callback(self, joint_torque):
		self.joint_torque[36:42] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])

	def joint_8_callback(self, joint_torque):
		self.joint_torque[42:48] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])

	def joint_9_callback(self, joint_torque):
		self.joint_torque[48:54] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])

	def joint_10_callback(self, joint_torque):
		self.joint_torque[54:60] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])


	def joint_11_callback(self, joint_torque):
		self.joint_torque[60:66] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])


	def joint_12_callback(self, joint_torque):
		self.joint_torque[66:72] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])

	def joint_13_callback(self, joint_torque):
		self.joint_torque[72:78] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])

	def joint_14_callback(self, joint_torque):
		self.joint_torque[78:84] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])


	def joint_15_callback(self, joint_torque):
		self.joint_torque[84:90] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])


	def joint_16_callback(self, joint_torque):
		self.joint_torque[90:96] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])


	def joint_17_callback(self, joint_torque):
		self.joint_torque[96:102] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])

	def joint_18_callback(self, joint_torque):
		self.joint_torque[102:108] = np.asarray([joint_torque.wrench.force.x, joint_torque.wrench.force.y, joint_torque.wrench.force.z, joint_torque.wrench.torque.x, joint_torque.wrench.torque.y, joint_torque.wrench.torque.z])

def listener(current_robot_state):

    rospy.init_node('state_listener', anonymous=True)

    rospy.Subscriber("/snake_monster/sensors/imu", Imu,  current_robot_state.imu_callback)
    rospy.Subscriber("/snake_monster/sensors/SA001__MoJo/torque", WrenchStamped,  current_robot_state.joint_1_callback)
    rospy.Subscriber("/snake_monster/sensors/SA002__MoJo/torque", WrenchStamped,  current_robot_state.joint_2_callback)
    rospy.Subscriber("/snake_monster/sensors/SA003__MoJo/torque", WrenchStamped,  current_robot_state.joint_3_callback)
    rospy.Subscriber("/snake_monster/sensors/SA005__MoJo/torque", WrenchStamped,  current_robot_state.joint_4_callback)
    rospy.Subscriber("/snake_monster/sensors/SA0013__MoJo/torque", WrenchStamped,  current_robot_state.joint_5_callback)
    rospy.Subscriber("/snake_monster/sensors/SA0014__MoJo/torque", WrenchStamped,  current_robot_state.joint_6_callback)
    rospy.Subscriber("/snake_monster/sensors/SA0022__MoJo/torque", WrenchStamped,  current_robot_state.joint_7_callback)
    rospy.Subscriber("/snake_monster/sensors/SA0023__MoJo/torque", WrenchStamped,  current_robot_state.joint_8_callback)
    rospy.Subscriber("/snake_monster/sensors/SA0030__MoJo/torque", WrenchStamped,  current_robot_state.joint_9_callback)
    rospy.Subscriber("/snake_monster/sensors/SA0031__MoJo/torque", WrenchStamped,  current_robot_state.joint_10_callback)
    rospy.Subscriber("/snake_monster/sensors/SA0036__MoJo/torque", WrenchStamped,  current_robot_state.joint_11_callback)
    rospy.Subscriber("/snake_monster/sensors/SA0037__MoJo/torque", WrenchStamped,  current_robot_state.joint_12_callback)
    rospy.Subscriber("/snake_monster/sensors/SA0038__MoJo/torque", WrenchStamped,  current_robot_state.joint_13_callback)
    rospy.Subscriber("/snake_monster/sensors/SA0043__MoJo/torque", WrenchStamped,  current_robot_state.joint_14_callback)
    rospy.Subscriber("/snake_monster/sensors/SA0046__MoJo/torque", WrenchStamped,  current_robot_state.joint_15_callback)
    rospy.Subscriber("/snake_monster/sensors/SA0048__MoJo/torque", WrenchStamped,  current_robot_state.joint_16_callback)
    rospy.Subscriber("/snake_monster/sensors/SA0049__MoJo/torque", WrenchStamped,  current_robot_state.joint_17_callback)
    rospy.Subscriber("/snake_monster/sensors/SA0050__MoJo/torque", WrenchStamped,  current_robot_state.joint_18_callback)



    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    current_robot_state = robot_state()
    listener(current_robot_state)
