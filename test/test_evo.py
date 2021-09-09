#!/usr/bin/env python3

from griddly import GymWrapper, gd
import numpy as np
import torch
import copy as cp
from tqdm import tqdm
from coevo import AgentNet, get_state, Individual, Canonical, AgentInd, EnvInd, define_action_space, play_one_game


#np.random.seed(0)
#actions aléatoires

def init_custom_env(pretty_mode=False, level=0, game="simple_maze"):
    if pretty_mode:
        env = GymWrapper(yaml_file=game+".yaml",
            global_observer_type=gd.ObserverType.SPRITE_2D, player_observer_type=gd.ObserverType.SPRITE_2D, level=level)
    else:
        env = GymWrapper(yaml_file=game+".yaml", level=level)

    return env


def test_agent_es():
    envInd = EnvInd()
    env = envInd.env
    obs = env.reset()
    n_action = env.action_space
    agent = AgentNet(get_state(obs), n_action)

    params = agent.get_params()
    d = len(params)

    es = Canonical(d)
    assert es.n_pop > 0 # population size

    for i in range(es.n_pop):
        es.population.append(AgentInd(env))

    pop = es.ask()
    orig_pop = cp.copy(pop)

    for i in pop:
        i.play_game(envInd, render = False)
        assert i.fitness >= -15

    print("max = ", max(pop, key=lambda p: p.fitness))

    es.tell(pop)

    pop2 = es.ask()

    assert len(orig_pop) == len(pop2)

    for i in range(len(pop)):
        assert np.any(orig_pop[i].genes != pop2[i].genes)

    for i in pop2:
        i.play_game(envInd)

    print("max = ", max(pop2, key=lambda p: p.fitness))


def test_generations():
    envInd = EnvInd()
    env = envInd.env
    obs = env.reset()
    n_action = env.action_space
    agent = AgentNet(get_state(obs), n_action)

    params = agent.get_params()
    d = len(params)
    es = Canonical(d)

    for i in range(es.n_pop):
        es.population.append(AgentInd(env))

    for i in range(3):
        print("-------- Iteration ", i+1," --------")
        pop = es.ask()

        for i in pop:
            i.play_game(envInd, render=False)

        es.tell(pop)
        es.log()

    print("max = ", max(pop, key=lambda p: p.fitness))

    #es.plot(data='mean')


def test_evolution_zelda():
    envInd = EnvInd()
    env = envInd.env
    obs = env.reset()
    agent = AgentNet(get_state(obs), env.action_space)

    params = agent.get_params()
    d = len(params)

    es = Canonical(d)
    assert es.n_pop > 0 # population size

    for i in range(es.n_pop):
        es.population.append(AgentInd(env=env))

    pop = es.ask()
    orig_pop = cp.copy(pop)

    for i in pop:
        i.play_game(envInd, render = False)
        assert i.fitness <= AgentInd.nb_steps_max + 30

    print("max = ", max(pop, key=lambda p: p.fitness))

    es.tell(pop)

    pop2 = es.ask()

    assert len(orig_pop) == len(pop2)

    for i in range(len(pop)):
        assert np.any(orig_pop[i].genes != pop2[i].genes)

    for i in pop2:
        i.play_game(envInd)

def test_generation_zelda():
    envInd = EnvInd()
    env = envInd.env
    #define_action_space(env)
    obs = env.reset()
    agent = AgentNet(get_state(obs), env.action_space)

    params = agent.get_params()
    d = len(params)

    i=0
    sum=0

    best_ind = None
    best_fitness = 0

    for i in tqdm(range(1000)):
        # print("-------- Iteration ", i+1," --------")
        if (i==0):
            envInd = EnvInd()
            es = Canonical(d)
            for j in range(es.n_pop):
                es.population.append(AgentInd(env=envInd.env))

        pop = es.ask()

        for k in pop:
            play_one_game(k, envInd)
            k.compute_fitness()

        es.tell(pop)
        maximum = max(pop, key=lambda p: p.fitness)
        if maximum.fitness > best_fitness:
            best_fitness = maximum.fitness
            best_ind = cp.copy(maximum.agent.state_dict())

        sum = sum + maximum.fitness
        print("max = ", maximum)


    # for i in range(es.n_pop):
    #     torch.save(pop[i].agent.state_dict(), "save/last_agent"+str(i))

    torch.save(best_ind, "save/best_agent_trained_normal_env")
    print("Best fitness = ", best_fitness)
    es.plot(data='max')

def test_save_agent():
    env = init_custom_env(game="simple_zelda")
    obs = env.reset()
    agent = AgentNet(get_state(obs), env.action_space)

    torch.save(agent.state_dict(), "test_save")

def test_load_agent(number=5):
    envInd = EnvInd()
    env = envInd.env
    define_action_space(env)
    obs = env.reset()
    agent = AgentNet(get_state(obs), env.action_space)

    agent.load_state_dict(torch.load("save/last_agent"+str(number)))

    indiv = AgentInd(env=env, genes=agent.get_params())
    indiv.play_game(envInd, render=False)

def load_best_agent():
    envInd = EnvInd()
    env = envInd.env
    define_action_space(env)
    obs = env.reset()
    agent = AgentNet(get_state(obs), env.action_space)

    agent.load_state_dict(torch.load("best_agent2"))

    indiv = AgentInd(env=env, genes=agent.get_params())
    indiv.play_game(envInd, render=False)
    print(indiv.fitness)



#test_agent_es()
#test_generations()
#test_evolution_zelda()
test_generation_zelda()
#load_best_agent()

#test_save_agent()
#test_load_agent(30)
