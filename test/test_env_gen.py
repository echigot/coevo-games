from coevo.canonical import Canonical
from torch._C import DisableTorchFunction
from coevo.individual import EnvInd, define_action_space
import gym
import pytest
import numpy as np
import copy as cp

from griddly import GymWrapperFactory, gd

from coevo import AgentInd




def test_gen_map():
    env_ind = EnvInd()
    agent_ind = AgentInd(env_ind.env)
    env_ind.play_game(agent_ind.agent)


def test_automaton():
    env_ind = EnvInd()
    
    old_grid = cp.copy(env_ind.CA.grid)

    for i in range(5):
        env_ind.evolve_CA()
    
    distance = np.linalg.norm(old_grid) - np.linalg.norm(env_ind.CA.grid)
    assert np.abs(distance) > 0


def test_evo_automaton():
    env_ind = EnvInd()

    len_params = len(env_ind.genes)
    
    es = Canonical(len_params, direction="min")
    for i in range(es.n_pop):
        es.population.append(EnvInd())
    
    for i in range(5):

        pop = es.ask()

        for j in pop:
            j.evolve_CA()
        
        es.tell(pop)
        maximum = max(pop, key=lambda p: p.fitness)
        #print(maximum.CA.grid)
        

test_evo_automaton()