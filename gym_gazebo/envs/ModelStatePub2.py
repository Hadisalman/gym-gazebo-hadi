#!/usr/bin/env python
import roslib; roslib.load_manifest('gazebo_ros')
import sys
from gazebo_msgs.msg import ModelState
import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState


def main():
	pub = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=10)
	rospy.init_node('dynamic_obstacles')
	rate = rospy.Rate(10)	

	g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
	rospy.wait_for_service("/gazebo/get_model_state")
	init_state = g_get_state(model_name="my_robot")
	init_state_0 = g_get_state(model_name="my_robot_0")
	init_state_1 = g_get_state(model_name="my_robot_1")
	init_state_2 = g_get_state(model_name="my_robot_2")
	init_state_3 = g_get_state(model_name="my_robot_3")

	# print(state.pose)

	state=ModelState()
	state.model_name = "my_robot"
	state.pose.position.x=init_state.pose.position.x
	state.pose.position.y=init_state.pose.position.y

	state0=ModelState()
	state0.model_name = "my_robot_0"
	state0.pose.position.x=init_state_0.pose.position.x
	state0.pose.position.y=init_state_0.pose.position.y

	state1=ModelState()
	state1.model_name = "my_robot_1"
	state1.pose.position.x=init_state_1.pose.position.x
	state1.pose.position.y=init_state_1.pose.position.y

	state2=ModelState()
	state2.model_name = "my_robot_2"
	state2.pose.position.x=init_state_2.pose.position.x
	state2.pose.position.y=init_state_2.pose.position.y

	state3=ModelState()
	state3.model_name = "my_robot_3"
	state3.pose.position.x=init_state_3.pose.position.x
	state3.pose.position.y=init_state_3.pose.position.y

	while not rospy.is_shutdown():
		i=0
		for j in range(60):
			state.pose.position.x=state.pose.position.x+i
			state0.pose.position.x=state0.pose.position.x+i
			state1.pose.position.x=state1.pose.position.x+i
			state2.pose.position.x=state2.pose.position.x+i
			state3.pose.position.x=state3.pose.position.x+i
		
			pub.publish(state)
			pub.publish(state0)
			pub.publish(state1)
			pub.publish(state2)
			pub.publish(state3)

			i=i+0.001
			rate.sleep()
		i=0	
		for j in range(60):
			state.pose.position.x=state.pose.position.x-i
			state0.pose.position.x=state0.pose.position.x-i
			state1.pose.position.x=state1.pose.position.x-i
			state2.pose.position.x=state2.pose.position.x-i
			state3.pose.position.x=state3.pose.position.x-i
		
			pub.publish(state)
			pub.publish(state0)
			pub.publish(state1)
			pub.publish(state2)
			pub.publish(state3)

			i=i+0.001
			rate.sleep()	
		
		# state.pose.orientation.z=0.5
		# for j in range(60):



if __name__ == '__main__':
	try:
		main()

	except rospy.ROSInterruptException:

		pass	