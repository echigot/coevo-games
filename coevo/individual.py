from coevo.agent_net import AgentNet
import gym
from griddly import GymWrapper, gd
import numpy as np
import torch
from coevo import get_state
import random as rd


class Individual:
    nb_steps_max = 7

    def __init__(self, genes):
        self.genes = genes
        self.fitness = 0
        self.done = False
        self.steps = 0
        self.age = 0

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, new_genes):
        self._genes = new_genes
        self.fitness = 0
        self.age = 0
        self.steps = 0
        self.done = False
        self.agent.set_params(new_genes)

    def __repr__(self):
        return f"ES Indiv (fitness={self.fitness})"

    def __str__(self):
        return self.__repr__()

    def do_action(self, result, env):
        obs_2, reward, done, info = env.step(result)
        self.done = done
        self.fitness = reward+self.fitness
        if (self.done):
            env.reset()
        return obs_2

    def play_one_game(self, agent, env, render=False):
        obs = env.reset()
        while((not self.done) and self.steps < Individual.nb_steps_max):
            #obs = np.random.randint(0,2,size=obs.shape)
            result = get_result(agent, obs)
            obs = self.do_action(result, env)
            self.steps = self.steps + 1
            if render:
                env.render()
            
        self.fitness = fitness(self, env)


class AgentInd(Individual):

    def __init__(self, env=None, genes=None):
        if env is None:
            env = GymWrapper(yaml_file="simple_maze.yaml", level=0)

        obs = env.reset()
        
        self.agent = AgentNet(get_state(obs), env.action_space.n)

        if genes is not None:
            self.genes = genes
            self.agent.set_params(genes)
        else:
            print(self.agent.get_params())
            #self.agent.set_params(self.agent.get_params())
            self.genes = self.agent.get_params()

        super(AgentInd, self).__init__(self.genes)

    def play_game(self, env, render=False):
        self.play_one_game(self.agent, env, render)

    


class EnvInd(Individual):

    def __init__(self, genes=None):
        if genes is None:
            # TODO: create genes
            genes = np.random.rand(1)
        super(EnvInd, self).__init__(genes)
        self.env = None

    def play_game(self, agent):
        self.play_one_game(agent, self.env)
            
    

def fitness(indiv, env):
    goal_location = get_object_location(env, 'exit')
    avatar_location = get_object_location(env, 'avatar')
    distance = np.linalg.norm(np.array(goal_location) - np.array(avatar_location))
    return indiv.fitness*10 - distance
    
def get_result(agent, obs):
    #print("state ",get_state(obs))
    actions = agent(get_state(obs))
    print(actions.detach().numpy())
    a = int(np.argmax(actions.detach().numpy()))
    #print(a)
    return a

def get_object_location(env, object):
    for i in env.get_state()["Objects"]:
        if (i['Name']==object):
            return i['Location']
    
    return None
