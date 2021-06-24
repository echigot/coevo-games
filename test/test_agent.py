#!/usr/bin/env python3

import gym
import griddly
import numpy as np
from coevo import AgentNet, get_state, Individual, fitness

def test_nn_creation():
    env = gym.make('GDY-Labyrinth-v0')
    obs = env.reset()

    n_channel = obs.shape[0]
    n_action = env.action_space.n

    agent = AgentNet(n_channel, n_action)
    result = agent.forward(get_state(obs))
    assert len(result) == 5

def test_fitness():
    env = gym.make('GDY-Labyrinth-v0')
    obs = env.reset()

    n_channel = obs.shape[0]
    n_action = env.action_space.n

    agent = AgentNet(n_channel, n_action)

    parameters = agent.get_parameters()
    indiv = Individual(parameters)
    assert indiv.fitness == 0
    assert np.all(indiv.genes == parameters)

    fit = fitness(indiv, env)

    assert fit >= 0 


