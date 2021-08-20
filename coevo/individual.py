from coevo.env_grid import EnvGrid
from matplotlib.pyplot import step
from coevo.agent_net import AgentNet
import gym
from griddly import GymWrapper, gd
import numpy as np
import torch
from coevo import get_state
import random as rd
import copy as cp


class Individual:
    nb_steps_max = 200
    dic_actions = {}

    def __init__(self, genes):
        self.age = 0
        self.genes = genes


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
        self.last_action = [0,0]

    def __repr__(self):
        return f"ES Indiv (fitness={self.fitness})"

    def __str__(self):
        return self.__repr__()

    def do_action(self, result, env):
        obs_2, reward, done, _ = env.step(result)
        self.done = done
        self.fitness = reward*30 + self.fitness
        if (self.done):
            obs_2 = env.reset()
        return obs_2

    def play_one_game(self, agent, env, render=False, level_string=None):
        if level_string is None:
            obs = env.reset()
        else:
            obs = self.env.reset(level_string=level_string)

        while((not self.done) and self.steps < Individual.nb_steps_max):
            if render:
                env.render()
            result = get_result(agent, obs)
            if result != self.last_action and result in Individual.dic_actions:
                self.fitness = self.fitness + 1
                if result[0] != 1:
                    self.fitness = self.fitness +1
            obs = self.do_action(result, env)
            self.steps = self.steps + 1
            self.last_action = result
            
            
        self.fitness = self.compute_fitness()
        env.close()


class AgentInd(Individual):

    def __init__(self, env=None, genes=None, age=1):
        if env is None:
            self.env = GymWrapper(yaml_file="simple_zelda.yaml", level=0)
        else:
            self.env=env
            
        #self.avatar_init_location = get_object_location(env, 'avatar')
        obs = self.env.reset()
        
        self.agent = AgentNet(get_state(obs), self.env.action_space)

        if genes is None:
            genes = self.agent.get_params()
        else:
            self.agent.set_params(genes)

        super(AgentInd, self).__init__(genes)

    def play_game(self, env, render=False):
        self.play_one_game(self.agent, env, render)

    def compute_fitness(self):
        #avatar_location = get_object_location(env, 'avatar')
        #distance = np.abs(np.linalg.norm(np.array(indiv.avatar_init_location) - np.array(avatar_location)))
        # fitness = 0
        # if self.fitness > 0:
        #     fitness = 2 - self.steps/Individual.nb_steps_max
        # elif self.fitness < 0:
        #     fitness = self.steps/Individual.nb_steps_max - 2
        return self.fitness

    


class EnvInd(Individual):
    height = 13
    width = 9
    default_env = GymWrapper(yaml_file='simple_zelda.yaml', level=0)
    default_obs = cp.copy(default_env.reset())

    def __init__(self, genes=None):
        
        self.env = cp.copy(EnvInd.default_env)
        self.age = 0
        self.CA = EnvGrid(width=EnvInd.width, height=EnvInd.height, num_actions=7)

        if genes is None:
            genes = self.CA.get_params()
        else:
            self.CA.set_params(genes)
        
        super(EnvInd, self).__init__(genes)

        self.fitness = self.compute_fitness()


    def play_game(self, agent, render=False):
        level_string = self.env_to_string()
        self.play_one_game(agent, self.env, level_string=level_string, render=render)

    
    def env_to_string(self):
        level_string = ''
        for i in range(EnvInd.width):
            for j in range(EnvInd.height):
                block = self.CA.grid[j][i]
                level_string+=match_block(block).ljust(4)
            level_string += '\n'
        
        return level_string

    def compute_fitness(self):
        obs_th = EnvInd.default_obs.astype(int)
        #print("default_obs = ")
        #print(obs_th)
        #print("------------")
        obs_gen = self.env.reset(level_string=self.env_to_string()).astype(int)
        #print(obs_gen)
        #distance = np.diff(obs_th - obs_gen, axis = -1)# - np.linalg.norm(obs_gen)
        distance = np.sum(np.abs(obs_th - obs_gen), axis=-1)
        #print(distance)
        distance = np.sum(distance)
        return distance

    def evolve_CA(self):
        self.CA.evolve()
        self.fitness = self.compute_fitness()

    
def match_block(x):
    return {
        0:'.',
        1:'A',
        2:'x',
        3:'+',
        4:'g',
        5:'3',
        6:'w',
    }[x]
            

def get_result(agent, obs):
    #print("state ",get_state(obs))
    actions = agent(get_state(obs)).detach().numpy()
    #print(actions)
    if (isinstance(agent.n_out, int)):
        a = int(np.argmax(actions))
    else:
        action = int(np.argmax(actions[0][:2]))
        direction = int(np.argmax(actions[0][2:]))
        a = (action, direction)
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

