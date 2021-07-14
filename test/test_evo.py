#!/usr/bin/env python3

import gym
from griddly import GymWrapper, gd
import numpy as np
from torch.functional import Tensor
import copy as cp
from coevo import AgentNet, get_state, Individual, fitness, Canonical, AgentInd, EnvInd


np.random.seed(0)
#implémenter zelda
#map 1234
#actions aléatoires

def init_custom_env(pretty_mode=False, level=0):
    if pretty_mode:
        env = GymWrapper(yaml_file="simple_maze.yaml",
            global_observer_type=gd.ObserverType.SPRITE_2D, player_observer_type=gd.ObserverType.SPRITE_2D, level=level)
    else:
        env = GymWrapper(yaml_file="simple_maze.yaml", level=level)
    
    return env





def test_agent_es():

    env = init_custom_env()
    obs = env.reset()
    n_action = env.action_space.n
    agent = AgentNet(get_state(obs), n_action)

    params = agent.get_params()
    d = len(params)

    es = Canonical(d)
    assert es.n_pop > 0 # population size

    for i in range(es.n_pop):
        es.population.append(AgentInd(env=env))

    pop = es.ask()
    orig_pop = cp.deepcopy(pop)

    for i in pop:
        i.play_game(env)
        assert i.fitness >= -15

    print("max = ", max(pop, key=lambda p: p.fitness))

    es.tell(pop)

    pop2 = es.ask()

    assert len(orig_pop) == len(pop2)

    for i in range(len(pop)):
        assert np.any(orig_pop[i].genes != pop2[i].genes)
    
    for i in pop2:
        i.play_game(env)

    print("max = ", max(pop2, key=lambda p: p.fitness))


def test_generations():
    env = init_custom_env()
    obs = env.reset()
    n_action = env.action_space.n
    agent = AgentNet(get_state(obs), n_action)

    params = agent.get_params()
    d = len(params)
    es = Canonical(d)

    for i in range(es.n_pop):
        es.population.append(AgentInd(env=env))


    env = init_custom_env()
    for i in range(10):
        print("-------- Iteration ", i+1," --------")
        pop = es.ask()
        
        for i in pop:
            i.play_game(env, render=True)

        es.tell(pop)
        es.log()
    
    print("max = ", max(pop, key=lambda p: p.fitness))

    # env = init_custom_env()
    # for i in range(0):
    #     print("-------- Iteration ", i+1," --------")
    #     pop = es.ask()
        
    #     for i in pop:
    #         i.play_game(env)

    #     es.tell(pop)
    #     es.log()
        
    # print("max = ", max(pop, key=lambda p: p.fitness))
    # env = init_custom_env()
    # for i in range(0):
    #     print("-------- Iteration ", i+1," --------")
    #     pop = es.ask()

    #     for i in pop:
    #         i.play_game(env, render=True)

    #     es.tell(pop)
    #     es.log()

    # print("max = ", max(pop, key=lambda p: p.fitness))

    es.plot(data='mean')

#test_agent_es()
test_generations()