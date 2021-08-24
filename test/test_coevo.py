#!/usr/bin/env python3

from coevo.population import PopEnv, PopInd
import gym
from griddly import GymWrapper, gd
import numpy as np
import torch
import copy as cp
from coevo import Canonical, AgentInd, EnvInd, Population


def test_coevolution():

    pop_env = PopEnv()
    pop_agents = PopInd()
    best_env = pop_env.get_best_ind()
    #pop_agents.improve(best_env)

    for i in range(100):
        pop_env.improve()

        pop_env.eliminate()
        print(pop_env.real_n_pop)
        
        pop_env.play(pop_agents)
        pop_agents.evaluate()
        pop_env.evaluate()

        best_env = pop_env.get_best_ind()
        best_agent = pop_agents.get_best_ind()
        
        print(best_agent)
        print(best_agent.fitness)

        print(best_env)
        print(best_env.env_to_string())

        pop_agents.evolve()
        pop_env.evolve()



test_coevolution()