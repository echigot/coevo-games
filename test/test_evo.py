#!/usr/bin/env python3

import gym
from griddly import GymWrapper, gd
import numpy as np
import torch
from torch.functional import Tensor
import copy as cp
from coevo import AgentNet, get_state, Individual, fitness, Canonical, AgentInd, EnvInd


np.random.seed(0)
#actions aléatoires

def init_custom_env(pretty_mode=False, level=0, game="simple_maze"):
    if pretty_mode:
        env = GymWrapper(yaml_file=game+".yaml",
            global_observer_type=gd.ObserverType.SPRITE_2D, player_observer_type=gd.ObserverType.SPRITE_2D, level=level)
    else:
        env = GymWrapper(yaml_file=game+".yaml", level=level)
    
    return env


def test_agent_es():

    env = init_custom_env()
    obs = env.reset()
    n_action = env.action_space
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
        i.play_game(env, render = True)
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
    for i in range(3):
        print("-------- Iteration ", i+1," --------")
        pop = es.ask()
        
        for i in pop:
            i.play_game(env, render=True)

        es.tell(pop)
        es.log()
    
    print("max = ", max(pop, key=lambda p: p.fitness))

    es.plot(data='mean')


def test_evolution_zelda():
    env = init_custom_env(game="simple_zelda")
    obs = env.reset()
    agent = AgentNet(get_state(obs), env.action_space)

    params = agent.get_params()
    d = len(params)

    es = Canonical(d)
    assert es.n_pop > 0 # population size

    for i in range(es.n_pop):
        es.population.append(AgentInd(env=env))

    pop = es.ask()
    orig_pop = cp.deepcopy(pop)

    for i in pop:
        i.play_game(env, render = True)
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

def test_generation_zelda():
    env = init_custom_env(game="simple_zelda")
    obs = env.reset()
    agent = AgentNet(get_state(obs), env.action_space)

    params = agent.get_params()
    d = len(params)
    es = Canonical(d)


    env = init_custom_env(game="simple_zelda")
    i=0
    sum=0
    render=False

    best_ind = None
    best_fitness = 0

    while (i<50):
        print("-------- Iteration ", i+1," --------")
        if (i==0):
            env = init_custom_env(game="simple_zelda")
            es = Canonical(d)
            for j in range(es.n_pop):
                es.population.append(AgentInd(env=env))

        if (i>=4 and sum < 7):
            i=0
            sum=0
            continue

        pop = es.ask()
        
        for k in pop:
            k.play_game(env, render=render)

        es.tell(pop)
        es.log()
        maximum = max(pop, key=lambda p: p.fitness)
        if maximum.fitness > best_fitness:
            best_fitness = maximum.fitness
            best_ind = cp.deepcopy(maximum.agent.state_dict())
            
        sum = sum + maximum.fitness
        print("max = ", maximum)

        i = i+1
    
    torch.save(best_ind, "best_agent")
    es.plot(data='mean')
    es = None
    env = None
    pop = None

def test_save_agent():
    env = init_custom_env(game="simple_zelda")
    obs = env.reset()
    agent = AgentNet(get_state(obs), env.action_space)

    torch.save(agent.state_dict(), "test_save")

def test_load_agent():
    env = init_custom_env(game="simple_zelda")
    obs = env.reset()
    agent = AgentNet(get_state(obs), env.action_space)

    agent.load_state_dict(torch.load("test_save"))

    indiv = AgentInd(env=env, genes=agent.get_params())
    indiv.play_game(env, render=True)

def load_best_agent():
    env = init_custom_env(game="simple_zelda")
    obs = env.reset()
    agent = AgentNet(get_state(obs), env.action_space)

    agent.load_state_dict(torch.load("best_agent"))

    indiv = AgentInd(env=env, genes=agent.get_params())
    indiv.play_game(env, render=True)
    print(indiv.fitness)



#test_agent_es()
#test_generations()
#test_evolution_zelda()
#test_generation_zelda()
load_best_agent()

#test_save_agent()
#test_load_agent()