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

def normalize(mean, std, value):
    # TODO
    return np.divide(value, std) - mean

def unnormalize(mean_deltas, std_deltas, state_diff):
    # TODO
    return mean_deltas + np.multiply(std_deltas, state_diff)

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
        obs_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        # TODO: what does input placeholder look like
        self.input_ph = tf.placeholder([None, obs_dim + ac_dim],  name="ob_ac", dtype=tf.float32)
        self.state_diff_ph = tf.placeholder([None, obs_dim], name="state_diff", dtype=tf.float32)
        # TODO: output_size?
        self.out = build_mlp(self.input_ph, obs_dim, 'train', n_layers, size, activation, output_activation)
        self.normalization = normalization
        self.learning_rate = learning_rate
        self.sess = sess
        self.batch_size = batch_size
        self.env = env
        self.loss = tf.losses.mean_squared_error(self.state_diff_ph, self.out)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        """YOUR CODE HERE """
        # TODO: get batch & figure out what is input placeholder look like, normalize
        for _ in range(iterations):
            obs_batch = 
            next_obs_batch = 
            ac_batch = 
            state_diff_batch = next_obs_batch - obs_batch
            feed_dict = {
                self.input_ph: np.append(obs_batch, ac_batch),
                self.state_diff_ph: state_diff_batch
            }
            self.sess.run(self.train_op, feed_dict=feed_dict)


    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        # TODO: figure out what input place holder looks like, unnormalize
        normalized_states, normalized_actions = normalize(states), normalize(actions)
        feed_dict = {
            self.input_ph: np.append(obs_batch, ac_batch)
        }
        state_diff_batch = self.sess.run(self.out, feed_dict=feed_dict)
        return states + unnormalize(state_diff_batch)
