#!/usr/bin/env python3

from coevo.individual import AgentInd
import gym
from griddly import GymWrapper, gd
import numpy as np
from torch.functional import Tensor
from coevo import AgentNet, get_state, Individual, fitness

def test_nn_creation():
    env = gym.make('GDY-Labyrinth-v0')
    obs = env.reset()
    n_action = env.action_space.n
    agent = AgentNet(obs, n_action)

    result = agent.forward(get_state(obs))

    assert Tensor.size(result)[1] == n_action

def test_fitness():
    env = gym.make('GDY-Labyrinth-v0')  
    
    indiv = AgentInd(env=env)
    assert len(indiv.agent.get_params()) >= 0
    assert indiv.fitness == 0
    assert np.all(indiv.genes == indiv.agent.get_params())
    assert indiv.done == False

    fit = fitness(indiv, env)

    assert fit <= 0

def test_fitness_after_step():
    env = GymWrapper(yaml_file="simple_maze.yaml")
    
    indiv = AgentInd(env)
    assert indiv.fitness == 0

    fit_init = fitness(indiv, env)
    assert fit_init >=-15

    indiv.play_game(env)

    current_fit = fitness(indiv, env)
    assert current_fit >=-15

def test_agent_zelda():
    env = GymWrapper(yaml_file='simple_zelda.yaml', global_observer_type=gd.ObserverType.SPRITE_2D, player_observer_type=gd.ObserverType.SPRITE_2D)
    env.reset()
    indiv = AgentInd(env)
    indiv.play_game(env, render=True)

test_agent_zelda()