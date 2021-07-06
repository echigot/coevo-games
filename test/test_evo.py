#!/usr/bin/env python3

import gym
from griddly import GymWrapper, gd
import numpy as np
from torch.functional import Tensor
from coevo import AgentNet, get_state, Individual, fitness, EvoStrat, Canonical

def init_test_env():
    env = gym.make('GDY-Labyrinth-v0')
    obs = env.reset()
    n_action = env.action_space.n
    agent = AgentNet(obs, n_action)
    return env, obs, n_action, agent


def test_agent_es():

    env, obs, n_action, agent = init_test_env()
    params = agent.get_parameters()
    d = len(params)

    es = Canonical(d)
    assert es.n > 0 # population size

    for i in range(es.n):
        es.population.append(AgentInd())

    pop = es.ask()
    orig_pop = deepcopy(pop)

    for i in pop:
        i.play_game(env)
        assert i.fitness <= 0

    es.tell(pop)

    es.update()

    pop2 = es.ask()

    assert len(orig_pop) == len(pop2)

    for i in range(len(pop)):
        assert np.any(orig_pop[i].genes != pop2[i].genes)


def test_population():

    population = []

    for i in range(10):
        population.append(Individual())

    for e in population:
        print("--")
        obs = e.env.reset()
        n_action = e.env.action_space.n
        e.indiv.play_one_game(e.agent, e.env, obs, n_action)

    best_indiv = EvoStrat.select(population, 3)
    new_pop = EvoStrat.evolve(best_indiv, len(population))
