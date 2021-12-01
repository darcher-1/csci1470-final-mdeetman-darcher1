import os
import random
import cv2
import gym
import numpy as np
import tensorflow as tf

EPSILON = 0.05

def train(env):
    pass

#TODO compile 4 previous frames
def preprocess(img):
    img = cv2.resize(img, (84,110))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[26:110,:]
    cv2.imshow("Input",img)
    cv2.waitKey(40)
    

def main():
    env = gym.make("Breakout-v0")
    num_actions = env.action_space.n
    preprocess(env.reset())
    done = False
    while not done:
        action = np.zeros(num_actions, dtype='uint8')
        action = env.action_space.sample()
        if (random.uniform(0,1) > EPSILON):
            #TODO: use Q function from model
            action = env.action_space.sample()
        state, _, done, _ = env.step(action)
        preprocess(state)

if __name__ == '__main__':
    main()

