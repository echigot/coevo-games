#!/usr/bin/env python3

import gym
from griddly import GymWrapperFactory, gd
import numpy as np
import torch
from torch.functional import Tensor
from torch.nn import parameter
from coevo import AgentNet, get_state, Individual, fitness



def init_test_env():
    env = gym.make('GDY-Labyrinth-v0')
    obs = env.reset()
    n_action = env.action_space.n
    agent = AgentNet(obs, n_action)
    return env, obs, n_action, agent

def test_nn_creation():
    env, obs, n_action, agent = init_test_env()

    result = agent.forward(get_state(obs))

    assert Tensor.size(result)[1] == 5

def test_fitness():
    env, obs, n_action, agent = init_test_env()

    parameters = agent.get_parameters()

    assert parameters != None
    
    indiv = Individual(parameters)
    assert indiv.fitness == 0
    assert np.all(indiv.genes == parameters)
    assert indiv.done == False

    fit = fitness(indiv, env)

    assert fit >= 0

def test_fitness_after_step():
    env, obs, n_action, agent = init_test_env()

    parameters = agent.get_parameters()
    
    indiv = Individual(parameters)
    assert indiv.fitness == 0

    fit_init = fitness(indiv, env)
    assert fit_init == 0

    #indiv.play_one_game(agent, env, obs, n_action)

    current_fit = fitness(indiv, env)
    assert current_fit >= fit_init




def test_population():
    wrapper = GymWrapperFactory()
    wrapper.build_gym_from_yaml('SimpleMaze', 'simple_maze.yaml')
    env = gym.make('GDY-SimpleMaze-v0')

    obs = env.reset()
    n_action = env.action_space.n
    agent = AgentNet(obs, n_action)

    parameters = agent.get_parameters()
    population = []
    for i in range(10):
        population.append(Individual(parameters))




#with torch.no_grad():
#    params = agent.parameters()
#    vec = torch.nn.utils.parameters_to_vector(params)
#    print( len(vec.cpu().numpy()))