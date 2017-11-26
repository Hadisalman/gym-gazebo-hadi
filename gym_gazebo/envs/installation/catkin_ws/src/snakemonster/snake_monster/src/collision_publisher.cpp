
//Author: Saurabh Nair; snnair@andrew.cmu.edu
//Date: Nov 20th 2017

#include <ros/ros.h>

// MoveIt!
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene/planning_scene.h>

#include<snake_monster/check_hit.h>


bool check_hit_func(snake_monster::check_hit::Request  &req, snake_monster::check_hit::Response &res)
{
	
	robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
	robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();
	ROS_INFO("Model frame: %s", kinematic_model->getModelFrame().c_str());

	planning_scene::PlanningScene planning_scene(kinematic_model);


	collision_detection::CollisionRequest collision_request;
	collision_detection::CollisionResult collision_result;
	collision_result.clear();
	
	planning_scene.checkSelfCollision(collision_request, collision_result);
	//planning_scene.checkCollision(collision_request, collision_result);
	ROS_INFO_STREAM("Test 1: Current state is "
			<< (collision_result.collision ? "in" : "not in")
			<< " self collision");

	collision_detection::CollisionResult::ContactMap::const_iterator it;
  	it = collision_result.contacts.begin();
	ROS_INFO("Contact between: %s and %s",
           it->first.first.c_str(),
           it->first.second.c_str());
	for(it = collision_result.contacts.begin();
    		it != collision_result.contacts.end();
    		++it)
	{	
  	ROS_INFO("Contact between: %s and %s",
           it->first.first.c_str(),
           it->first.second.c_str());
	}

	
	res.status = collision_result.collision;
  	return true;
}


int main(int argc, char **argv)
{
	ros::init (argc, argv, "snake_monster_kinematics");

	ros::NodeHandle n;
//	ros::AsyncSpinner spinner(1);
//	spinner.start();

	
  	ros::ServiceServer service = n.advertiseService("check_hit", check_hit_func);
  	ROS_INFO("Ready to check self-collision");
	

	ros::spin();
	return 0;


}
