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
	rate = rospy.Rate(70)	

	g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
	rospy.wait_for_service("/gazebo/get_model_state")
	init_state = g_get_state(model_name="my_robot")
	init_state_0 = g_get_state(model_name="my_robot_0")
	init_state_1 = g_get_state(model_name="my_robot_1")
	init_state_2 = g_get_state(model_name="my_robot_2")
	init_state_3 = g_get_state(model_name="my_robot_3")
	init_state_4 = g_get_state(model_name="my_robot_4")
	init_state_5 = g_get_state(model_name="my_robot_5")
	init_state_6 = g_get_state(model_name="my_robot_6")
	init_state_7 = g_get_state(model_name="my_robot_7")
	init_state_8 = g_get_state(model_name="turtlebot")
	init_state_9 = g_get_state(model_name="turtlebot_0")
	init_state_10 = g_get_state(model_name="turtlebot_1")
	init_state_11 = g_get_state(model_name="turtlebot_2")
	init_state_12 = g_get_state(model_name="turtlebot_3")

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

	state4=ModelState()
	state4.model_name = "my_robot_4"
	state4.pose.position.x=init_state_4.pose.position.x
	state4.pose.position.y=init_state_4.pose.position.y

	state5=ModelState()
	state5.model_name = "my_robot_5"
	state5.pose.position.x=init_state_5.pose.position.x
	state5.pose.position.y=init_state_5.pose.position.y

	state6=ModelState()
	state6.model_name = "my_robot_6"
	state6.pose.position.x=init_state_6.pose.position.x
	state6.pose.position.y=init_state_6.pose.position.y

	state7=ModelState()
	state7.model_name = "my_robot_7"
	state7.pose.position.x=init_state_7.pose.position.x
	state7.pose.position.y=init_state_7.pose.position.y

	state8=ModelState()
	state8.model_name = "turtlebot"
	state8.pose.position.x=init_state_8.pose.position.x
	state8.pose.position.y=init_state_8.pose.position.y

	state9=ModelState()
	state9.model_name = "turtlebot_0"
	state9.pose.position.x=init_state_9.pose.position.x
	state9.pose.position.y=init_state_9.pose.position.y

	state10=ModelState()
	state10.model_name = "turtlebot_1"
	state10.pose.position.x=init_state_10.pose.position.x
	state10.pose.position.y=init_state_10.pose.position.y

	state11=ModelState()
	state11.model_name = "turtlebot_2"
	state11.pose.position.x=init_state_11.pose.position.x
	state11.pose.position.y=init_state_11.pose.position.y

	state12=ModelState()
	state12.model_name = "turtlebot_3"
	state12.pose.position.x=init_state_12.pose.position.x
	state12.pose.position.y=init_state_12.pose.position.y

	while not rospy.is_shutdown():
		i=0
		for j in range(60):
			state.pose.position.x=state.pose.position.x+i
			state0.pose.position.x=state0.pose.position.x+i
			state1.pose.position.x=state1.pose.position.x+i
			state2.pose.position.x=state2.pose.position.x+i
			state3.pose.position.x=state3.pose.position.x+i
			state4.pose.position.x=state4.pose.position.x+i
			state5.pose.position.x=state5.pose.position.x+i
			state6.pose.position.x=state6.pose.position.x+i
			state7.pose.position.x=state7.pose.position.x+i

			state8.pose.position.x=state8.pose.position.x+i
			state9.pose.position.x=state8.pose.position.x+i
			state10.pose.position.x=state10.pose.position.x+i
			state11.pose.position.y=state11.pose.position.y+i
			state12.pose.position.x=state12.pose.position.x+i
			
			state2.pose.position.y=state2.pose.position.y+i
			state3.pose.position.y=state3.pose.position.y+i

			pub.publish(state)
			pub.publish(state0)
			pub.publish(state1)
			pub.publish(state2)
			pub.publish(state3)
			pub.publish(state4)
			pub.publish(state5)
			pub.publish(state6)
			pub.publish(state7)
			pub.publish(state8)
			pub.publish(state9)
			pub.publish(state10)
			pub.publish(state11)
			pub.publish(state12)

			i=i+0.001
			rate.sleep()
		i=0	
		for j in range(60):
			state.pose.position.x=state.pose.position.x-i
			state0.pose.position.x=state0.pose.position.x-i
			state1.pose.position.x=state1.pose.position.x-i
			state2.pose.position.x=state2.pose.position.x-i
			state3.pose.position.x=state3.pose.position.x-i
			state4.pose.position.x=state4.pose.position.x-i
			state5.pose.position.x=state5.pose.position.x-i
			state6.pose.position.x=state6.pose.position.x-i
			state7.pose.position.x=state7.pose.position.x-i

			state8.pose.position.x=state8.pose.position.x-i
			state9.pose.position.x=state8.pose.position.x-i
			state10.pose.position.x=state10.pose.position.x-i
			state11.pose.position.y=state11.pose.position.y-i
			state12.pose.position.x=state12.pose.position.x-i
		
			state2.pose.position.y=state2.pose.position.y-i
			state3.pose.position.y=state3.pose.position.y-i

			pub.publish(state)
			pub.publish(state0)
			pub.publish(state1)
			pub.publish(state2)
			pub.publish(state3)
			pub.publish(state4)
			pub.publish(state5)
			pub.publish(state6)
			pub.publish(state7)
			pub.publish(state8)
			pub.publish(state9)
			pub.publish(state10)
			pub.publish(state11)
			pub.publish(state12)

			i=i+0.001
			rate.sleep()	
		
		# state.pose.orientation.z=0.5
		# for j in range(60):



if __name__ == '__main__':
	try:
		main()

	except rospy.ROSInterruptException:

		pass	