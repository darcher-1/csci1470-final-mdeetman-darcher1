import os
import sys
import random
import cv2
import gym
import numpy as np
import tensorflow as tf
from collections import deque
from Preprocess import Preprocess
from FrameBuffer import FrameBuffer
from DQN import DQNModel
import matplotlib.pyplot as plt

EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999
GAMMA = 0.99
TARGET_UPDATE_STEPS = 5
NUM_EPISODES = 900
MIN_REPLAY_SIZE = 1000
MAX_REPLAY_SIZE = 50000
MINIBATCH_SIZE = 64
LOAD_WEIGHTS = False

class DQNAgent():
    def __init__(self, num_actions):
        self.model = DQNModel(num_actions, (84,84,4))
        if (LOAD_WEIGHTS):
            self.model.load_weights('weights')
        self.target_model = DQNModel(num_actions, (84,84,4))
        self.target_model.set_weights(self.model.get_weights())
        self.target_model_counter = 0
        self.replay_memory = deque(maxlen=MAX_REPLAY_SIZE)
        self.epsilon = 1.

    def train(self, done, num_actions):
        if len(self.replay_memory) < MIN_REPLAY_SIZE:
            return False
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        states = np.array([step[0] for step in minibatch])
        actions = np.array([step[1] for step in minibatch])
        rewards = np.array([step[2] for step in minibatch])
        target_states = np.array([step[3] for step in minibatch])
        done_batch = np.array([step[4] for step in minibatch])
        not_done_batch = 1 - done_batch
        with tf.GradientTape() as tape:
            prbs = self.model.call(states)
            #prob corresponding to chosen acion calculated below
            prbs = tf.reduce_sum(tf.one_hot(actions, num_actions) * prbs, axis=1)
            target_prbs = tf.reduce_max(self.target_model.call(target_states), axis=-1)
            y = rewards + GAMMA * target_prbs * not_done_batch
            loss = tf.keras.losses.mean_squared_error(y, prbs)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        if (done):
            self.target_model_counter += 1
            if (self.target_model_counter >= TARGET_UPDATE_STEPS):
                self.target_model.set_weights(self.model.get_weights())
                self.target_model_counter = 0
                print('target network updated')
                self.model.save_weights('weights')
        return True

    def update_replay_mem(self, step):
        self.replay_memory.append(step)

    def predict(self, state):
        return self.model.call(state)

    def decayEpsilon(self):
        self.epsilon *= EPSILON_DECAY
    
def play_without_train(env, agent):
    while True:
        done = False
        state = env.reset()
        total_reward = 0
        while not done:
            action = np.argmax(agent.predict(state.reshape(1,84,84,4)))
            print(action)
            state, _, done, _ = env.step(action)
            if len(sys.argv) > 1:
                if sys.argv[1] == "RENDER":
                    env.render()
        print("total reward over last episode:", total_reward, "with", steps, "steps")

def main():
    env = gym.make("BreakoutDeterministic-v4")
    env = Preprocess(env)
    env = FrameBuffer(env)
    agent = DQNAgent(env.action_space.n)
    if LOAD_WEIGHTS:
        play_without_train(env, agent)
        return
    for i in range(NUM_EPISODES):
        steps = 1
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()
            if (random.uniform(0,1) > max(agent.epsilon, EPSILON_MIN)):
                action = np.argmax(agent.predict(state.reshape(1,84,84,4)))
            agent.decayEpsilon()
            new_state, reward, done, _ = env.step(action)
            steps += 1
            total_reward += reward;
            agent.update_replay_mem((state, action, reward, new_state, done))
            state = new_state
            training = agent.train(done, env.action_space.n)
            if len(sys.argv) > 1:
                if sys.argv[1] == "RENDER":
                    env.render()
        print("total reward over episode " + str(i+1) + ":", total_reward, "with", steps, "steps.", "Epsilon is", round(max(agent.epsilon, EPSILON_MIN), 2))
    

if __name__ == '__main__':
    main()
