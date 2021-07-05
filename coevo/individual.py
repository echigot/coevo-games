import gym
import griddly
import numpy as np
import torch
from coevo import get_state

class Individual:
    nb_steps_max = 15

    def __init__(self, genes):
        self.genes = genes
        self.fitness = 0
        self.done = False
        self.steps = 0

    def do_action(self, result, env):
        obs_2, reward, done, info  = env.step(result)
        self.done = done
        self.fitness = reward+self.fitness
        if (self.done):
            env.reset()
        return obs_2
          
    def play_one_game(self, agent, env, obs, n_action):
        #obs = env.reset()
        while((not self.done) and self.steps < Individual.nb_steps_max):
            result = get_result(agent, obs)
            obs = self.do_action(result, env)
            #print(result)
            #env.render()
            self.steps = self.steps + 1

        self.fitness = fitness(self, env)

            
    

def fitness(indiv, env):
    goal_location = get_object_location(env, 'exit')
    avatar_location = get_object_location(env, 'avatar')
    distance = np.linalg.norm(np.array(goal_location) - np.array(avatar_location))
    print('distance = ' , distance)
    return indiv.fitness - distance/Individual.nb_steps_max
    
def get_result(agent, obs):
    actions = agent(get_state(obs))
    a = int(np.argmax(actions.detach().numpy()))
    return a

def get_object_location(env, object):
    for i in env.get_state()["Objects"]:
        if (i['Name']==object):
            return i['Location']
    
    return None