import numpy as np
import tensorflow as tf
import gym
from dynamics import NNDynamicsModel
from controllers import MPCcontroller, RandomController
from cost_functions import cheetah_cost_fn, trajectory_cost_fn
import time
import logz
import os
import copy
import matplotlib.pyplot as plt
from cheetah_env import HalfCheetahEnvNew
import pickle

def sample(env, 
           controller, 
           num_paths=10, 
           horizon=1000, 
           render=False,
           verbose=False):
    """
        Write a sampler function which takes in an environment, a controller (either random or the MPC controller), 
        and returns rollouts by running on the env. 
        Each path can have elements for observations, next_observations, rewards, returns, actions, etc.
    """
    paths = []
    """ YOUR CODE HERE """
    for i in range(num_paths): 
        print(i)
        observation = env.reset() 
        observations = [observation] 
        actions = []
        rewards = []
        for i in range(horizon):
            action = controller.get_action(observation)
            observation, reward, done, info = env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
        path = {'observations': np.array(observations[:-1]),
            'actions': np.array(actions),
            'next_observations': np.array(observations[1:]),
            'rewards': np.array(rewards)}
        paths.append(path)

    # paths: {'observations': [horizon, num_paths, obs/ac/rew_dim], ...}
    # import pdb; pdb.set_trace()
    return paths

# Utility to compute cost a path for a given cost function
def path_cost(cost_fn, path):
    return trajectory_cost_fn(cost_fn, path['observations'], path['actions'], path['next_observations'])

# def append_paths(paths, new_paths):
#     paths['observations'] = np.append(paths['observations'], new_paths['observations'], axis=1)
#     paths['actions'] = np.append(paths['actions'], new_paths['actions'], axis=1)
#     paths['next_observations'] = np.append(paths['next_observations'], new_paths['next_observations'], axis=1)
#     paths['rewards'] = np.append(paths['rewards'], new_paths['rewards'], axis=1)

def compute_normalization(data):
    """
    Write a function to take in a dataset and compute the means, and stds.
    Return 6 elements: mean of s_t, std of s_t, mean of (s_t+1 - s_t), std of (s_t+1 - s_t), mean of actions, std of actions
    """
    """ YOUR CODE HERE """
    # data has shape [horizon, num_paths, obs/ac_dim]
    obs, ac, next_obs = data
    mean_obs, std_obs = np.mean(obs, axis = 0), np.std(obs, axis = 0)
    mean_action, std_action = np.mean(ac, axis = 0), np.std(ac, axis = 0)
    deltas = next_obs - obs
    mean_deltas, std_deltas = np.mean(deltas, axis = 0), np.std(deltas, axis = 0)
    return mean_obs, std_obs, mean_action, std_action, mean_deltas, std_deltas

def get_data_from_paths(paths):
    observations = []
    actions = []
    next_observations = []

    for path in paths:
        observations.append(path['observations'])
        actions.append(path['actions'])
        next_observations.append(path['next_observations'])
    obs = np.array(observations)
    ac = np.array(actions)
    next_obs = np.array(next_observations)
    obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2])
    ac = ac.reshape(ac.shape[0] * ac.shape[1], ac.shape[2])
    next_obs = next_obs.reshape(next_obs.shape[0] * next_obs.shape[1], next_obs.shape[2])
    return obs, ac, next_obs


def plot_comparison(env, dyn_model):
    """
    Write a function to generate plots comparing the behavior of the model predictions for each element of the state to the actual ground truth, using randomly sampled actions. 
    """
    """ YOUR CODE HERE """
    states = [env.reset()]
    predicted_states = [env.reset()]
    for _ in range(100):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        states.append(state)
        predicted_states.append(dyn_model.predict(state, action))
    

def train(env, 
         cost_fn,
         logdir=None,
         render=False,
         learning_rate=1e-3,
         onpol_iters=10,
         dynamics_iters=60,
         batch_size=512,
         num_paths_random=10, 
         num_paths_onpol=10, 
         num_simulated_paths=10000,
         env_horizon=1000, 
         mpc_horizon=15,
         n_layers=2,
         size=500,
         activation=tf.nn.relu,
         output_activation=None
         ):

    """

    Arguments:

    onpol_iters                 Number of iterations of onpolicy aggregation for the loop to run. 

    dynamics_iters              Number of iterations of training for the dynamics model
    |_                          which happen per iteration of the aggregation loop.

    batch_size                  Batch size for dynamics training.

    num_paths_random            Number of paths/trajectories/rollouts generated 
    |                           by a random agent. We use these to train our 
    |_                          initial dynamics model.
    
    num_paths_onpol             Number of paths to collect at each iteration of
    |_                          aggregation, using the Model Predictive Control policy.

    num_simulated_paths         How many fictitious rollouts the MPC policy
    |                           should generate each time it is asked for an
    |_                          action.

    env_horizon                 Number of timesteps in each path.

    mpc_horizon                 The MPC policy generates actions by imagining 
    |                           fictitious rollouts, and picking the first action
    |                           of the best fictitious rollout. This argument is
    |                           how many timesteps should be in each fictitious
    |_                          rollout.

    n_layers/size/activations   Neural network architecture arguments. 

    """

    logz.configure_output_dir(logdir)

    #========================================================
    # 
    # First, we need a lot of data generated by a random
    # agent, with which we'll begin to train our dynamics
    # model.

    random_controller = RandomController(env)

    """ YOUR CODE HERE """
    paths = sample(env, random_controller, num_paths_random, env_horizon)
    costs = np.array([path_cost(cost_fn, path) for path in paths])
    returns = np.array([np.sum(path['rewards']) for path in paths])
    #========================================================
    # 
    # The random data will be used to get statistics (mean
    # and std) for the observations, actions, and deltas
    # (where deltas are o_{t+1} - o_t). These will be used
    # for normalizing inputs and denormalizing outputs
    # from the dynamics network. 
    # 
    data = get_data_from_paths(paths)
    normalization = compute_normalization(data)
    # for path in paths:
    #     path['observations'] = np.divide((path['observations'] - mean_obs), std_obs)


    #========================================================
    # 
    # Build dynamics model and MPC controllers.
    # 
    sess = tf.Session()

    dyn_model = NNDynamicsModel(env=env, 
                                n_layers=n_layers, 
                                size=size, 
                                activation=activation, 
                                output_activation=output_activation, 
                                normalization=normalization,
                                batch_size=batch_size,
                                iterations=dynamics_iters,
                                learning_rate=learning_rate,
                                sess=sess)

    mpc_controller = MPCcontroller(env=env, 
                                   dyn_model=dyn_model, 
                                   horizon=mpc_horizon, 
                                   cost_fn=cost_fn, 
                                   num_simulated_paths=num_simulated_paths)


    #========================================================
    # 
    # Tensorflow session building.
    # 
    sess.__enter__()
    tf.global_variables_initializer().run()

    #========================================================
    # 
    # Take multiple iterations of onpolicy aggregation at each iteration refitting the dynamics model to current dataset and then taking onpolicy samples and aggregating to the dataset. 
    # Note: You don't need to use a mixing ratio in this assignment for new and old data as described in https://arxiv.org/abs/1708.02596
    # 
    avg_returns = []
    for itr in range(onpol_iters+1):
        """ YOUR CODE HERE """
        data = get_data_from_paths(paths)
        dyn_model.fit(data)
        paths_onpol = sample(env, mpc_controller, num_paths_onpol, env_horizon) 
	paths += paths_onpol
        costs = np.array([path_cost(cost_fn, path) for path in paths])
        returns = np.array([np.sum(path['rewards']) for path in paths])
        avg_returns.append(np.mean(returns))


        # LOGGING
        # Statistics for performance of MPC policy using
        # our learned dynamics model
        logz.log_tabular('Iteration', itr)
        # In terms of cost function which your MPC controller uses to plan
        logz.log_tabular('AverageCost', np.mean(costs))
        logz.log_tabular('StdCost', np.std(costs))
        logz.log_tabular('MinimumCost', np.min(costs))
        logz.log_tabular('MaximumCost', np.max(costs))
        # In terms of true environment reward of your rolled out trajectory using the MPC controller
        logz.log_tabular('AverageReturn', np.mean(returns))
        logz.log_tabular('StdReturn', np.std(returns))
        logz.log_tabular('MinimumReturn', np.min(returns))
        logz.log_tabular('MaximumReturn', np.max(returns))

        logz.dump_tabular()
    pickle.dump(avg_returns, open("avg_returns.p", "wb"))

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    # Experiment meta-params
    parser.add_argument('--exp_name', type=str, default='mb_mpc')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--render', action='store_true')
    # Training args
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--onpol_iters', '-n', type=int, default=15)
    parser.add_argument('--dyn_iters', '-nd', type=int, default=60)
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    # Data collection
    parser.add_argument('--random_paths', '-r', type=int, default=10)
    parser.add_argument('--onpol_paths', '-d', type=int, default=10)
    parser.add_argument('--simulated_paths', '-sp', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=int, default=1000)
    # Neural network architecture args
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=500)
    # MPC Controller
    parser.add_argument('--mpc_horizon', '-m', type=int, default=15)
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Make data directory if it does not already exist
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    # Make env
    if args.env_name is "HalfCheetah-v1":
        env = HalfCheetahEnvNew()
        cost_fn = cheetah_cost_fn
    train(env=env, 
                 cost_fn=cost_fn,
                 logdir=logdir,
                 render=args.render,
                 learning_rate=args.learning_rate,
                 onpol_iters=args.onpol_iters,
                 dynamics_iters=args.dyn_iters,
                 batch_size=args.batch_size,
                 num_paths_random=args.random_paths, 
                 num_paths_onpol=args.onpol_paths, 
                 num_simulated_paths=args.simulated_paths,
                 env_horizon=args.ep_len, 
                 mpc_horizon=args.mpc_horizon,
                 n_layers = args.n_layers,
                 size=args.size,
                 activation=tf.nn.relu,
                 output_activation=None,
                 )

if __name__ == "__main__":
    main()
