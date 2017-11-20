class reward_function:
  

    def __init__(self, state_1, state_2, action_taken, dt, start_state, goal_state):
        self.current_state = s2
        self.previous_state = s1
        self.action = action_taken
        self.goal_state = goal_state
        self.start_state = start_state

    #State is arranged as:
    #Joint angles(18), Joint velocities(18), Joint Torques(),  Force-feedback(), IMU pose(3DOF +  4quat), bot pose(7)
    current_state
    previous_state
    def forward_acceleration(self):

	#This threshold has to be decided!
	snap_point = 2

        forward_acceleration = np.sqrt(np.square(state_2.imu_state[8]) + np.square(state_2.imu_state[9]))       
        slow_down_point = np.norm(self.current_state.robot_state[0:3], self.goal_state[0:3]) > snap_point

	if(slow_down_point):
		reward_acceleration = 10*forward_acceleration
	else:
		reward_acceleration = -10*forward_acceleration #Promote deceleration at point close to the goal!

        return reward_acceleration

    #Not sure why I added this :p
    def any_movement(self):
       self. 
         
    #Check for ground contact for each leg 
    def ground_contact(self):
        if(np.sum(self.current_state.end_effector_z[0:6] < 0.2) >= 2):
            return 100
	else
	    return -10
        
    #Periodicity model
    def conservation_of_energy(self):
        	
    
    #polygon of support
    def static_stability(self):
		

    #Fucking add a penalty if the robot hits itself
    def self_collision(self):
        if(self.current_state.collision):
           return -1000
        else:
            return 10
        
    #To have minimum energy added to the system
    def control_input(self):
        #The action should be the difference in the state joint angles? Or just the torques 
        input_energy = self.action*self.action.T
        reward_control_input = (10.0/input_energy)
	return reward_control_input        

    #Add reward for being normal to the surface
    def slip_avoidance(self):
        return 1


