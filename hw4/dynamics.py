import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.obs_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        # TODO: what does input placeholder look like
        self.input_ph = tf.placeholder(shape=[None, self.obs_dim + self.ac_dim],  name="ob_ac", dtype=tf.float32)
        self.delta_ph = tf.placeholder(shape=[None, self.obs_dim], name="state_diff", dtype=tf.float32)
        # TODO: output_size?
        self.out = build_mlp(self.input_ph, self.obs_dim, 'train', n_layers, size, activation, output_activation)
        self.mean_obs, self.std_obs, self.mean_ac, self.std_ac, self.mean_deltas, self.std_deltas = normalization
        self.learning_rate = learning_rate
        self.sess = sess
        self.batch_size = batch_size
        self.env = env
        self.loss = tf.losses.mean_squared_error(self.delta_ph, self.out)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.iterations = iterations

    def normalize(self, obs, ac, delta):
        # takes in batches of obs, ac, delta, and returns normalized obs, acs, and deltas:
        epsilon = 1e-15
        normalized_obs = np.divide(obs - self.mean_obs, self.std_obs + epsilon)
        normalized_ac = np.divide(ac - self.mean_ac, self.std_ac + epsilon)
        normalized_delta = np.divide(delta - self.mean_deltas, self.std_deltas + epsilon)
        return normalized_obs, normalized_ac, normalized_delta

    def normalize_inference(self, obs, ac):
        epsilon = 1e-15
        normalized_obs = np.divide(obs - self.mean_obs, self.std_obs + epsilon)
        normalized_ac = np.divide(ac - self.mean_ac, self.std_ac + epsilon)
        return normalized_obs, normalized_ac

    def unnormalize(self, delta):
        return np.multiply(delta, self.std_deltas) + self.mean_deltas

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        """YOUR CODE HERE """
        # TODO: get batch & figure out what is input placeholder look like, normalize
        # data = {'observations': [horizon, num_paths, obs/ac_dim]}
        
        # flatten paths data across horizon and number of paths
        train_obs, train_ac, train_next_obs = data
        for _ in range(self.iterations):
	    indices = np.arange(train_obs.shape[0])
	    np.random.shuffle(indices) 
	    train_obs, train_ac, train_next_obs = train_obs[indices], train_ac[indices], train_next_obs[indices]
            for cur_batch in range(train_obs.shape[0] / self.batch_size):
            # shape: [batch_size, obs/ac_dim]
                obs_batch = train_obs[cur_batch * self.batch_size : (cur_batch+1) * self.batch_size]
                next_obs_batch = train_next_obs[cur_batch * self.batch_size : (cur_batch+1) * self.batch_size]
                ac_batch = train_ac[cur_batch * self.batch_size : (cur_batch+1) * self.batch_size]
                delta_batch = next_obs_batch - obs_batch
                
                obs_batch, ac_batch, delta_batch = self.normalize(obs_batch, ac_batch, delta_batch)
                feed_dict = {
                    self.input_ph: np.append(obs_batch, ac_batch, axis = 1),
                    self.delta_ph: delta_batch
                }
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)


    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        # TODO: figure out what input place holder looks like, unnormalize
        normalized_states, normalized_actions = self.normalize_inference(states, actions)
        feed_dict = {
            self.input_ph: np.append(normalized_states, normalized_actions, axis = 1)
        }
        delta_batch = self.sess.run(self.out, feed_dict=feed_dict)
        return states + self.unnormalize(delta_batch)
