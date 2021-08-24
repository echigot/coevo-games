from coevo.individual import AgentInd, EnvInd, Individual
from coevo.canonical import Canonical
import numpy as np


class Population:
    def __init__(self, indType, direction):
        self.indType = indType

        init_ind = indType()
        len_params = len(init_ind.genes)

        self.es = Canonical(len_params, direction=direction) 

        for i in range(self.es.n_pop):
            self.es.population.append(indType())

        self.pop = self.es.population
        self.real_n_pop = self.es.n_pop

    def evaluate(self):
        for i in self.pop:
            i.compute_fitness()

    def evolve(self):
        self.es.tell(self.pop)
        self.pop = self.es.ask()
        self.real_n_pop = self.es.n_pop
    
    def get_best_ind(self):
        return max(self.pop, key=lambda p: p.fitness)


class PopEnv(Population):

    def __init__(self, direction="max"):
        super().__init__(EnvInd, direction=direction)

    def eliminate(self):
        for i in self.pop:
            if i.remove_bad_env():
                self.real_n_pop = self.real_n_pop - 1
    
    def play(self, pop_agents):
        for a in (pop_agents.pop):
            for e in (self.pop):
                if e.playable:
                    e.play_game(a)

    def improve(self):
        for k in range(np.sqrt(EnvInd.width*EnvInd.height).astype(int)):
            for j in self.pop:
                j.evolve_CA()




class PopInd(Population):

    def __init__(self, direction="max"):
        super().__init__(AgentInd, direction=direction)

    def improve(self, env):
        pop = self.pop
        for i in range(5):
            
            for j in pop:
                j.play_game(env.env)
        
            self.es.tell(pop)
            pop = self.es.ask()