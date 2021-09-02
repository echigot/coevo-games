#!/usr/bin/env python3

from coevo import play_one_game
from sys import platform
from coevo.population import PopEnv, PopInd
import gym
from griddly import GymWrapper, gd
import numpy as np
import torch
import copy as cp
from tqdm import tqdm
from coevo import Canonical, AgentInd, EnvInd, Population


def test_coevolution():

    pop_env = PopEnv()
    pop_agents = PopInd()
    best_env = pop_env.get_best_ind()
    generic_env = EnvInd()

    pop_agents.improve(generic_env, 20)

    for i in tqdm(range(50)):
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
        
        #pop_agents.improve(generic_env, 10)

    pop_agents.plot()
    pop_env.plot()
    play_one_game(best_agent, best_env, video=True, title="video_last_1.mp4")

    agent_hof = AgentInd(genes=pop_agents.es.hof[0].genes)
    env_hof = EnvInd(genes=pop_env.es.hof[0].genes)
    play_one_game(agent_hof, env_hof, video=True, title="video_hof_1.mp4")
    
test_coevolution()