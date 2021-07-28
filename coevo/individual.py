from matplotlib.pyplot import step
from coevo.agent_net import AgentNet
import gym
from griddly import GymWrapper, gd
import numpy as np
import torch
from coevo import get_state
import random as rd


class Individual:
    nb_steps_max = 130
    dic_actions = {}

    def __init__(self, genes):
        self.age = -1
        self.genes = genes


    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, new_genes):
        self._genes = new_genes
        self.fitness = 0
        self.age = self.age + 1
        self.steps = 0
        self.done = False
        self.last_action = [0,0]
        if (self.age > 1):
            self.agent.set_params(new_genes)
            pass
    

    def __repr__(self):
        return f"ES Indiv (fitness={self.fitness})"

    def __str__(self):
        return self.__repr__()

    def do_action(self, result, env):
        obs_2, reward, done, _ = env.step(result)
        self.done = done
        self.fitness = reward*20 + self.fitness
        if (self.done):
            obs_2 = env.reset()
        return obs_2

    def play_one_game(self, agent, env, render=False):
        obs = env.reset()

        while((not self.done) and self.steps < Individual.nb_steps_max):
            result = get_result(agent, obs)
            if result != self.last_action and result in Individual.dic_actions:
                self.fitness = self.fitness + 1
            obs = self.do_action(result, env)
            self.steps = self.steps + 1
            self.last_action = result
            if render:
                env.render()
            
        self.fitness = fitness(self, env)
        env.close()

    


class AgentInd(Individual):

    def __init__(self, env=None, genes=None):
        if env is None:
            env = GymWrapper(yaml_file="simple_maze.yaml", level=0)
            
        self.avatar_init_location = get_object_location(env, 'avatar')
        obs = env.reset()
        
        self.agent = AgentNet(get_state(obs), env.action_space)

        if genes is None:
            genes = self.agent.get_params()

        super(AgentInd, self).__init__(genes)

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
    #avatar_location = get_object_location(env, 'avatar')
    #distance = np.abs(np.linalg.norm(np.array(indiv.avatar_init_location) - np.array(avatar_location)))
    return indiv.fitness
    
def get_result(agent, obs):
    #print("state ",get_state(obs))
    actions = agent(get_state(obs)).detach().numpy()
    #print(actions)
    action = int(np.argmax(actions[0][:2]))
    direction = int(np.argmax(actions[0][2:]))
    a = (action, direction)
    #print(a)
    return a

def get_object_location(env, object):
    for i in env.get_state()["Objects"]:
        if (i['Name']==object):
            return i['Location']
    
    return None


def define_action_space(env):
    env.reset()
    for s in range(1000):
        action = tuple(env.action_space.sample())
        if (action in Individual.dic_actions):
            Individual.dic_actions[action] = Individual.dic_actions[action] +1
        else:
            Individual.dic_actions[action] = 1

