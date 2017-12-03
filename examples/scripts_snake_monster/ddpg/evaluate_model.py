import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
import tflearn
"""
Gym gazebo for snake monster imports
"""
import rospy
import pprint as pp
import gym
import gym_gazebo
import time
import random
import time
import matplotlib
import matplotlib.pyplot as plt

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

max_steps = 1000
if __name__ == '__main__':
    #checkpoint_file = tf.train.latest_checkpoint('./results/checkpoints_ddpg_run2')
    checkpoint_file = tf.train.latest_checkpoint('/home/saurabh/underactuated_project/gym-gazebo-hadi/examples/scripts_snake_monster/ddpg/results/checkpoints_ddpg_run2')
    
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
	    outdir = '/tmp/gazebo_gym_experiments'
	    env = gym.make('GazeboCustomSnakeMonsterDDPG-v0')
	    env = gym.wrappers.Monitor(env, directory=outdir, force=True, write_upon_reset=True)	
            s_dim = env.observation_space.shape[1]
	    action_dim = env.action_space.shape[1]
	    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
	    print ('session restored')
	    
	    # Get the placeholders from the graph by name
            inputs = graph.get_operation_by_name("actor_inputs/X").outputs[0]
	    #inputs = tflearn.input_data(shape=[None, 185],name='actor_inputs')		
	    # Tensors we want to evaluate
            scaled_out = graph.get_operation_by_name("actor_sacled_out").outputs[0]
	    print ("restored the scaled outputs")
		
	    #start the simulation
	    s = env.reset()
            ep_reward = 0
            ep_ave_max_q = 0
	    actor_noise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(action_dim))
            for i in range(max_steps):

            	# Added exploration noise
            	#a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            	# Execute the action and get feedback
            	a = sess.run(scaled_out, feed_dict={inputs:np.reshape(s, (1, s_dim))}) + actor_noise()
	    	print a[0], "Check this value"	
            	s2, r, terminal, info = env.step(a[0])
		s = s2
