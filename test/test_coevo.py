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
import matplotlib.pyplot as plt
from coevo import Canonical, AgentInd, EnvInd, Population, AgentNet, get_state


def test_coevolution():

    surviving_envs = []

    pop_env = PopEnv()
    pop_agents = PopInd()
    best_env = pop_env.get_best_ind()
    generic_env = EnvInd()

    #pop_agents.improve(generic_env, 20)

    for i in tqdm(range(100)):
        pop_env.improve()

        pop_env.eliminate()
        print(pop_env.real_n_pop)
        surviving_envs.append((pop_env.real_n_pop/pop_env.es.n_pop)*100)

        pop_env.play(pop_agents)
        pop_agents.evaluate()
        pop_env.evaluate()

        best_env = pop_env.get_best_ind()
        best_agent = pop_agents.get_best_ind()

        print(best_agent)
        print(best_agent.fitness)

        print(best_env)
        print(best_env.env_to_string())

        if i%50 == 0:
            pop_agents.save(best_agent, "save/agent_"+str(i))
            pop_env.save(best_env, "save/env_"+str(i))
            play_one_game(best_agent, best_env, video=True, title="video/video_"+str(i)+".mp4")

        pop_agents.evolve()
        pop_env.evolve()

        #pop_agents.improve(generic_env, 10)



    plt.plot(surviving_envs)
    plt.show()

    pop_agents.plot()
    pop_env.plot()
    play_one_game(best_agent, best_env, video=True, title="video/video_last.mp4")

def test_best_agents():
    envInd = EnvInd()
    env = envInd.env
    #define_action_space(env)
    obs = env.reset()
    agent = AgentNet(get_state(obs), env.action_space)

    agent.load_state_dict(torch.load("save/best_agent_trained_normal_env"))

    indiv = AgentInd(env=env, genes=agent.get_params())
    play_one_game(indiv, envInd)#, video=True, title="video/best_agent_classic_training_normal_env.mp4")
    indiv.compute_fitness()
    print(indiv.fitness)
    return indiv.fitness


#test_coevolution()
#test_best_agents()

results = []
for i in range(200):
    results.append(test_best_agents())
print(sum(results)/200)
plt.plot(results)
plt.show()