import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass

class RandomController(Controller):
	def __init__(self, env):
		""" YOUR CODE HERE """
		self.env = env

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Your code should randomly sample an action uniformly from the action space """
		return self.env.action_space.sample()


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """
		# Sample num_simulated_paths number of length horizon actions from env
			# Q: how do you sample actions without states
		# TODO: write code to sample actions
		actions = self.sample_actions() # shape: [horizon, num_simulated_paths, ac_dim]
		# broadcast s to be [num_simulated_paths, observation_dim]
		state = env.reset()
		states = np.tile(initial_state, [self.num_simulated_paths, 1])
		paths = [for _ in range(self.num_simulated_paths)]
		observations_h = []
		next_observations_h = []
		# Use dynamics model to generate s, s_0 is the initial states
		for i in range(horizon):
			observations_h.append(states)
			next_states = self.dyn_model.predict(states, actions[i])
			next_observations_h.append(next_states)
			states = next_states

		# evaluate cost for each imaginary rollouts using cost_fn
		observations_h = np.array(observations_h) # shape: [horizon, num_simulated_paths, obs/ac_dim]
		next_observations_h = np.array(next_observations_h)
		costs = self.cost_fn(observations_h, actions, next_observations_h) # shape: [num_simulated_paths]	

		# Pick the a_0 with the least cost
		best_trj = np.argmin(costs)
		return actions[0, best_trj]

