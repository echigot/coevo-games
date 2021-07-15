#!/usr/bin/env python3

import gym
from griddly import GymWrapper, gd
import numpy as np
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
    while (i<50):
        print("-------- Iteration ", i+1," --------")
        if (i==0):
            es = Canonical(d)
            for j in range(es.n_pop):
                es.population.append(AgentInd(env=env))

        if (i>=3 and sum<=6):
            i=0
            sum=0
            continue

        pop = es.ask()
        
        for k in pop:
            k.play_game(env, render=False)

        es.tell(pop)
        es.log()
        maximum = max(pop, key=lambda p: p.fitness) 
        sum = sum + maximum.fitness
        print("max = ", maximum)

        i = i+1
    
    es.plot(data='mean')

    best_indiv = max(pop, key=lambda p: p.fitness)
    Individual.nb_steps_max = 400
    agent_copie = AgentInd(env=env)
    agent_copie.genes = best_indiv.genes
    agent_copie.play_game(env, render=True)


#test_agent_es()
#test_generations()
#test_evolution_zelda()
test_generation_zelda()