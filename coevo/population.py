from coevo.individual import AgentInd, EnvInd, Individual
from coevo.canonical import Canonical
import numpy as np
import torch

# Manages populations of individuals in a coevolution context
class Population:
    def __init__(self, indType, direction):
        # indType is an Individual: EnvInd or AgentInd
        self.indType = indType

        # creates one individual to initialize a Canonical population
        init_ind = indType()
        len_params = len(init_ind.genes)
        self.es = Canonical(len_params, direction=direction) 

        for i in range(self.es.n_pop):
            self.es.population.append(indType())

        # population is now the canonical population
        self.pop = self.es.population

        # number of individuals in the population
        self.real_n_pop = self.es.n_pop

    # Computes the fitness of each individual
    def evaluate(self):
        for i in self.pop:
            i.compute_fitness()

    # Evolves the population with ES
    def evolve(self):
        self.es.tell(self.pop)
        self.pop = self.es.ask()
        self.real_n_pop = self.es.n_pop
    
    # Returns the best individual of the population
    # (ie: best fitness)
    def get_best_ind(self):
        return max(self.pop, key=lambda p: p.fitness)

    def plot(self):
        self.es.plot(data="max")


# Population of EnvInd
class PopEnv(Population):

    def __init__(self, direction="max"):
        super().__init__(EnvInd, direction=direction)

    # Removes bad environments from the pool and computes the new
    # size of the population
    def eliminate(self):
        for i in self.pop:
            if i.remove_bad_env():
                self.real_n_pop = self.real_n_pop - 1
    
    # Plays every combination of agent and playable environment
    def play(self, pop_agents):
        for a in (pop_agents.pop):
            for e in (self.pop):
                if e.playable:
                    e.play_game(a)

    # Evolves each environment to give every cell of its CA
    # a chance to change value
    def improve(self):
        for j in self.pop:
            for i in range(np.sqrt(EnvInd.width*EnvInd.height).astype(int)):
                j.evolve_CA()
            
            j.CA.grid[1][1] = 5 # puts an agent 

    
    def save(self, envInd):
        torch.save(envInd.CA.cell_net.state_dict(), "env_save")



# Population of AgentInd
class PopInd(Population):

    def __init__(self, direction="max"):
        super().__init__(AgentInd, direction=direction)

    # Evolves a population of agents n times for a given
    # environment
    def improve(self, envInd, n):
        pop = self.pop
        for i in range(n):
            
            for j in pop:
                j.play_game(envInd)
        
            self.es.tell(pop)
            pop = self.es.ask()

    def save(self, agentInd):
        torch.save(agentInd.agent.state_dict(), "agent_save")