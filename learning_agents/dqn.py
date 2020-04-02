# Import tensorflow keras api
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.losses import *

# Other imports
import numpy as np
import random


def get_network(resolution=32):

    ### ADD LAYERS TO MODEL ###

    # Input layer
    input = Input(shape=(resolution, 3))

    # Scale values from 0 to 1
    normalized = Lambda(lambda x: x / 255.0)(input)

    # First convolution layer
    layer_c1 = Convolution1D(
        filters=8,
        kernel_size=int(resolution / 4),
        strides=2,
        padding="same",
        activation="relu",
        kernel_initializer=VarianceScaling(scale=1.0)
    )(normalized)

    # Second convolution layer
    layer_c2 = Convolution1D(
        filters=16,
        kernel_size=int(resolution / 8),
        strides=2,
        padding="same",
        activation="relu",
        kernel_initializer=VarianceScaling(scale=1.0)
    )(layer_c1)

    # Flattening layer
    layer_flatten = Flatten()(layer_c2)

    # Dense layer
    layer_d1 = Dense(
        units=32, activation="relu",
        kernel_initializer=VarianceScaling(scale=1.0)
    )(layer_flatten)

    # Output layer
    output = Dense(
        units=6,
        kernel_initializer=VarianceScaling(scale=1.0)
    )(layer_d1)

    # Full model
    q_network = Model(input, output)

    ### COMPILE MODEL ###

    q_network.compile(loss=Huber(),
                      optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=["accuracy"])

    # Reutrn the compiled network
    return q_network


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = [None] * capacity
        self.capacity = capacity
        self.size = 0
        self.replace_index = 0

    # Add a new memory to the list
    def add(self, s, a, r, s_prime):
        # Add memory to correct location in list
        self.memory[self.replace_index] = [s, a, r, s_prime]

        # Increment location in list where next memory goes to
        self.replace_index = (self.replace_index + 1) % self.capacity

        # Number of stored memories
        self.size = min((self.size + 1), self.capacity)

    # Return a random batch of memories
    def get_batch(self, batch_size):
        # Random selection of memories
        idx = [random.randint(0, self.size - 1) for _ in range(batch_size)]

        # Package all the info from the batch of memories
        states = np.stack([self.memory[i][0] for i in idx]).astype(np.uint8)
        actions = [self.memory[i][1] for i in idx]
        rewards = np.array([self.memory[i][2] for i in idx])
        next_states = np.stack([self.memory[i][3] for i in idx]).astype(np.uint8)

        return states, actions, rewards, next_states

        # DQN-based learning agent


class DQNAgent:

    def __init__(self, memsize, gamma, resolution=32):
        # Compile network
        self.q_net = get_network(resolution=resolution)

        # Save params
        self.gamma = gamma
        self.resolution = resolution

        # Attach replay memory
        self.memory = ReplayMemory(memsize)

        # So we don't have to create this every time
        self.num_actions = 6
        self.ones_mask = np.array([[1.0 for i in range(self.num_actions)]])

    # Convert state into format expected by network
    # Input should be a numpy array where each row is a state
    def preprocess_state(self, s):
        # Make sure s is a numpy array with correct dimensions
        s_np = np.array(s).astype(np.float32).reshape(1, self.resolution, 3)
        return s_np

    def preprocess_states(self, states):
        s_np = np.array(states).astype(np.float32)
        s_np = np.reshape(s_np, (s_np.shape[0], self.resolution, 3))
        return s_np

    # Return the best action for a given state
    # Also returns predicted Q value for the best action
    def choose_action(self, s):
        output = self.q_net.predict(self.preprocess_state(s))
        action = np.argmax(output)
        return action, output[action]

    # Train q network using a batch of memories
    def learn_from_memory(self, batch_size=32):

        # Get batch data
        states, actions, rewards, next_states = self.memory.get_batch(batch_size)

        # Preprocess inputs to network
        states_pp = self.preprocess_states(states)
        next_states_pp = self.preprocess_states(next_states)

        # Run current state through network
        y = self.q_net.predict(states_pp)
        mask = one_hot_encode(actions, self.num_actions)
        y_update = rewards + self.gamma * np.max(
            self.q_net.predict(next_states_pp), axis=1)
        y_update = np.tile(y_update, (self.num_actions, 1)).T
        y_update = np.multiply(mask, y_update)
        y = y - np.multiply(mask, y) + y_update

        # Fit the network to the batch
        self.q_net.train_on_batch(states_pp, y)

    def print_qnet_summary(self):
        print(self.q_net.summary())

    # Convenience function for one-hot encoding actions


def one_hot_encode(action_idx, num_actions):
    idx = np.array(action_idx)
    out = np.zeros(shape=(len(action_idx), num_actions))
    out[np.arange(len(action_idx)), idx] = 1.0

    return out