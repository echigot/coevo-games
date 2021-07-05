#!/usr/bin/env python3

from coevo.evo_strat import Element
import gym
from griddly import GymWrapper, gd
import numpy as np
from torch.functional import Tensor
from coevo import AgentNet, get_state, Individual, fitness, EvoStrat



def init_test_env():
    env = gym.make('GDY-Labyrinth-v0')
    obs = env.reset()
    n_action = env.action_space.n
    agent = AgentNet(obs, n_action)
    return env, obs, n_action, agent

def test_nn_creation():
    env, obs, n_action, agent = init_test_env()

    result = agent.forward(get_state(obs))

    assert Tensor.size(result)[1] == 5

def test_fitness():
    env, obs, n_action, agent = init_test_env()

    parameters = agent.get_parameters()

    assert parameters != None
    
    indiv = Individual(parameters)
    assert indiv.fitness == 0
    assert np.all(indiv.genes == parameters)
    assert indiv.done == False

    fit = fitness(indiv, env)

    assert fit <= 0

def test_fitness_after_step():
    env = GymWrapper(yaml_file="simple_maze.yaml")
    obs = env.reset()

    n_action = env.action_space.n
    agent = AgentNet(obs, n_action)

    parameters = agent.get_parameters()
    
    indiv = Individual(parameters)
    assert indiv.fitness == 0

    fit_init = fitness(indiv, env)
    assert fit_init <= 0

    indiv.play_one_game(agent, env, obs, n_action)

    current_fit = fitness(indiv, env)
    #assert current_fit != fit_init


def test_population():
    
    population = []

    for i in range(10):
        population.append(create_element())
    
    for e in population:
        print("--")
        obs = e.env.reset()
        n_action = e.env.action_space.n
        e.indiv.play_one_game(e.agent, e.env, obs, n_action)

    best_indiv = EvoStrat.select(population, 3)
    new_pop = EvoStrat.evolve(best_indiv, len(population))


def create_element():
    env = GymWrapper(yaml_file="simple_maze.yaml")
    obs = env.reset()
    n_action = env.action_space.n

    agent = AgentNet(obs, n_action)
    parameters = agent.get_parameters()
    indiv = Individual(parameters)

    return Element(agent=agent, env=env, indiv=indiv)

#test_population()

#with torch.no_grad():
#    params = agent.parameters()
#    vec = torch.nn.utils.parameters_to_vector(params)
#    print( len(vec.cpu().numpy()))