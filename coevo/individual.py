import gym
import griddly
import numpy as np
import torch
from coevo import get_state

class Individual:
    nb_steps_max = 100

    def __init__(self, genes):
        self.genes = genes
        self.fitness = 0
        self.done = False
        self.reward = 0
        self.steps = 0

    def do_action(self, result, env):
        obs_2, reward, done, info  = env.step(result)
        self.done = done
        self.reward = reward+self.reward
        if (self.done):
            env.reset()
        return obs_2
          
    def play_one_game(self, agent, env, obs, n_action):
        #obs = env.reset()
        while(not self.done or self.steps < Individual.nb_steps_max):
            result = get_result(agent, obs)
            obs = self.do_action(result, env)
            #print(result)
            #env.render()
            self.steps = self.steps + 1
            
    

def fitness(agent, env):
    return agent.reward
    #return (Individual.nb_steps_max - agent.step) if agent.done else agent.reward
    
def get_result(agent, obs):
    actions = agent(get_state(obs))
    a = int(np.argmax(actions.detach().numpy()))
    return a
