#!/usr/bin/env python3

import gym
from griddly import GymWrapperFactory, gd


def test_griddly():
    env = gym.make('GDY-Labyrinth-v0')
    env.reset()

    for s in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())

        if done:
            env.reset()

def test_custom_env():
    wrapper = GymWrapperFactory()
    wrapper.build_gym_from_yaml('SimpleMaze', 'simple_maze.yaml')
    env = gym.make('GDY-SimpleMaze-v0')

    for s in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            env.reset()

