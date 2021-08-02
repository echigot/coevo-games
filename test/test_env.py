#!/usr/bin/env python3

import gym
from griddly import GymWrapper, gd
import yaml
from yaml.loader import Loader
import numpy as np


def test_griddly():
    env = gym.make('GDY-Labyrinth-v0')
    env.reset()

    for s in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())

        if done:
            env.reset()

def test_custom_env():
    env = GymWrapper(yaml_file='simple_maze.yaml')
    env.reset()

    for s in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            env.reset()

def test_zelda_env():
    #env = gym.make('GDY-Zelda-v0')
    env = GymWrapper(yaml_file='simple_zelda.yaml')
    env.reset()

    for s in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()
