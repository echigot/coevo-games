import gym
import griddly

class Individual:
    nb_steps_max = 300

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
          
    def play_one_game(self, agent, env):
        cpt = 0
        while(not self.done or self.steps < Individual.nb_steps_max):
            result = agent.get_result(env) #find how to get a result with nn
            self.do_action(result, env)
            self.steps = self.steps + 1
    

def fitness(agent, env):
    #add distance to ending point maybe
    return (Individual.nb_steps_max - agent.step) if agent.done else agent.reward
    
