#!/usr/bin/env python3

import gym
from griddly import GymWrapper, gd
import yaml
from yaml.loader import Loader


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


#test_custom_env()