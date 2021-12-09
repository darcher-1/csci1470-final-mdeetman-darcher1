import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DQNModel(tf.keras.Model):
    def __init__(self, num_actions, state_shape):
        super(DQNModel, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.network = tf.keras.Sequential()
        self.network.add(tf.keras.layers.Conv2D(16, (8,8), strides=4, activation='relu', input_shape=state_shape))
        self.network.add(tf.keras.layers.Conv2D(32, (4,4), strides=2, activation='relu'))
        self.network.add(tf.keras.layers.Flatten())
        self.network.add(tf.keras.layers.Dense(256, activation='relu'))
        self.network.add(tf.keras.layers.Dense(num_actions, activation='linear'))

    def call(self, state):
        return self.network(state)



