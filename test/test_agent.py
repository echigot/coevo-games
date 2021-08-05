#!/usr/bin/env python3

from coevo.individual import AgentInd
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

    fit = indiv.compute_fitness()

    assert fit <= 0

def test_fitness_after_step():
    env = GymWrapper(yaml_file="simple_maze.yaml")
    
    indiv = AgentInd(env)
    assert indiv.fitness == 0

    fit_init = indiv.compute_fitness()
    assert fit_init >=-15

    indiv.play_game(env)

    current_fit = indiv.compute_fitness()
    assert current_fit >=-15

def test_agent_zelda():
    env = GymWrapper(yaml_file='simple_zelda.yaml', global_observer_type=gd.ObserverType.SPRITE_2D, player_observer_type=gd.ObserverType.SPRITE_2D)
    env.reset()
    indiv = AgentInd(env)
    indiv.play_game(env)

    assert indiv.fitness <= AgentInd.nb_steps_max + 30
