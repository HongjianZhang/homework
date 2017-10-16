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

	def sample_actions(self):
		return np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high,
			size=(self.horizon, self.num_simulated_paths, self.env.action_space.shape[0]))

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """
		# Sample num_simulated_paths number of length horizon actions from env
		actions = self.sample_actions() # shape: [horizon, num_simulated_paths, ac_dim]
		obs_dim = state.shape[0]
		# broadcast s to be [num_simulated_paths, observation_dim]
		observations = np.broadcast_to(state, (self.num_simulated_paths, obs_dim))
		observations_h = [observations]
		# Use dynamics model to generate s, s_0 is the initial states
		for i in range(self.horizon):
			observations = self.dyn_model.predict(observations, actions[i])
			observations_h.append(observations)

		# evaluate cost for each imaginary rollouts using cost_fn
		observations_h = np.array(observations_h) # shape: [horizon, num_simulated_paths, obs/ac_dim]
		costs = trajectory_cost_fn(self.cost_fn, observations_h[:-1], actions, observations_h[1:]) # shape: [num_simulated_paths]	

		# Pick the a_0 with the least cost
		best_trj = np.argmin(costs)
		return actions[0, best_trj]

