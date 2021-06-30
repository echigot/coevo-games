import re
from unittest import result
import gym
import griddly

class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = 0
        self.done = False
        self.reward = 0

    def do_action(self, result, env):
        obs_2, reward, done, info  = env.step(result)
        self.done = done
        self.reward = reward+self.reward
        if (self.done):
            env.reset()
        
    
    def play_one_game(self, agent, env):
        cpt = 0
        while(not self.done and cpt < 300):
            result = agent.get_result(env) #find how to get a result with nn
            self.do_action(result, env)
            cpt = cpt + 1
    

def fitness(agent, env):
    return agent.reward*10 if agent.done else agent.reward
    
