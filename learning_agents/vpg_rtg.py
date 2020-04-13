
# Disable excessive info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import packages
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as inits
import tensorflow.keras.activations as activations
import numpy as np



def get_network(pixels, channels, num_actions, lr):

    # Input layer - 4 inputs to match env
    inputs = tf.keras.Input(shape=(pixels, channels))

    # Scale values from 0 to 1
    normalized = layers.Lambda(lambda x: x / 255.0)(inputs)

    # First convolution layer
    layer_c1 = layers.Convolution1D(
        filters=10,
        kernel_size=int(pixels / 4),
        strides=2,
        padding="same",
        activation=activations.relu,
        kernel_initializer=inits.VarianceScaling(scale=1.0)
    )(normalized)

    # Second convolution layer
    layer_c2 = layers.Convolution1D(
        filters=20,
        kernel_size=int(pixels / 8),
        strides=2,
        padding="same",
        activation=activations.relu,
        kernel_initializer=inits.VarianceScaling(scale=1.0)
    )(layer_c1)

    # Flattening layer
    layer_flatten = layers.Flatten()(layer_c2)

    # Dense layer
    layer_d1 = layers.Dense(
        units=50,
        activation=activations.relu,
        kernel_initializer=inits.VarianceScaling(scale=1.0)
    )(layer_flatten)

    # Output layer
    outputs = layers.Dense(
        units=num_actions,
        activation=activations.softmax,
        kernel_initializer=inits.VarianceScaling(scale=1.0)
    )(layer_d1)

    # Connect model
    policy_net = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(lr=lr)

    # Loss function whose gradient is the policy gradient
    # A: output from the network
    # Y: 2D tensor (row 1 = actions, row 2 = returns) for each input state
    def compute_loss(y, policy_output):

        # Extract data from Y
        acts = tf.cast(tf.gather(y, 0), tf.int32)
        rets = tf.cast(tf.gather(y, 1), tf.float32)

        # Get the probs for the actions taken
        mask = tf.one_hot(acts, num_actions)
        prob = tf.reduce_sum(policy_output * mask, axis=1)

        # Compute loss and return
        log_prob = tf.math.log(prob)
        return -tf.reduce_mean(log_prob * rets)

    # Compile the model
    policy_net.compile(optimizer=optimizer, loss=compute_loss)

    # Return model
    return policy_net


class VpgRtgAgent:

    def __init__(self, pixels, channels, num_actions, policy_lr):

        # Compile network
        self.policy_network = get_network(pixels, channels, num_actions, policy_lr)

        self.pixels = pixels
        self.channels = channels
        self.num_actions = num_actions

    # Feed observation through policy network to get probabilities for actions
    def get_policy(self, obs):
        return self.policy_network.predict(obs)[0]

    # Feed observation through policy network and sample an action from output
    def get_action(self, obs):
        return int(np.random.choice(self.num_actions, 1, p=self.get_policy(obs)))

    #
    def train(self, obs, acts, weights):

        A = tf.convert_to_tensor(obs, dtype=tf.float32)
        Y1 = tf.convert_to_tensor(acts, dtype=tf.float32)
        Y2 = tf.convert_to_tensor(weights, dtype=tf.float32)
        Y = tf.stack([Y1, Y2])

        # Update network
        self.policy_network.train_on_batch(A, Y)


def compute_ret_2_go(rew_arr):

    rets_2_go = np.zeros(len(rew_arr))
    for i in range(len(rew_arr)):
        rets_2_go[:i+1] += rew_arr[i]

    return rets_2_go.tolist()






