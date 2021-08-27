#!/usr/bin/env python3

from coevo.individual import AgentInd, EnvInd
import gym
from griddly import GymWrapper, gd
import numpy as np
from torch.functional import Tensor
from coevo import AgentNet, get_state, Individual

def test_nn_creation():
    env = gym.make('GDY-Labyrinth-v0')
    obs = env.reset()
    n_action = env.action_space.n
    agent = AgentNet(get_state(obs), n_action)

    result = agent(get_state(obs)).detach().numpy().flatten()

    assert len(result) == n_action

def test_fitness():
    env = gym.make('GDY-Labyrinth-v0')  
    
    indiv = AgentInd(env=env)
    assert len(indiv.agent.get_params()) >= 0
    assert indiv.fitness == 0
    assert np.all(indiv.genes == indiv.agent.get_params())
    assert indiv.done == False

    indiv.compute_fitness()
    fit = indiv.fitness

    assert fit <= 0

def test_fitness_after_step():
    envInd = EnvInd()
    env = envInd.env
    
    indiv = AgentInd(env)
    assert indiv.fitness == 0

    indiv.compute_fitness()
    fit_init = indiv.fitness
    assert fit_init >=-20

    indiv.play_game(envInd)

    indiv.compute_fitness()
    current_fit = indiv.fitness
    assert current_fit >=-20

def test_agent_zelda():
    envInd = EnvInd()
    env = envInd.env
    env.reset()
    indiv = AgentInd(env)
    indiv.play_game(envInd)

    assert indiv.fitness <= AgentInd.nb_steps_max + 30
