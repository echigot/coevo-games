#!/usr/bin/env python3

import gym
import griddly


def test_griddly():
    env = gym.make('GDY-Labyrinth-v0')
    env.reset()

    # Replace with your own control algorithm!
    for s in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())

        #env.render(observer='global') # Renders the entire environment

        if done:
            env.reset()


