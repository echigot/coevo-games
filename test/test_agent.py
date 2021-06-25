#!/usr/bin/env python3

import gym
import griddly
import numpy as np
import torch
from torch.functional import Tensor
from coevo import AgentNet, get_state, Individual, fitness

def test_nn_creation():
    env = gym.make('GDY-Labyrinth-v0')
    obs = env.reset()

    n_action = env.action_space.n

    agent = AgentNet(obs, n_action)

    result = agent.forward(get_state(obs))

    assert Tensor.size(result)[1] == 5

def test_fitness():
    env = gym.make('GDY-Labyrinth-v0')
    obs = env.reset()

    n_action = env.action_space.n

    agent = AgentNet(obs, n_action)

    parameters = agent.get_parameters()
    assert parameters != None
    
    indiv = Individual(parameters)
    assert indiv.fitness == 0
    assert np.all(indiv.genes == parameters)

    fit = fitness(indiv, env)

    assert fit >= 0