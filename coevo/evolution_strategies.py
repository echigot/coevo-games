from coevo.individual import AgentInd
import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy, copy
import json


import gym


import platform
OS = platform.system()
import os
import errno

class ES_Ind:
    def __init__(self, genes):
        self._genes = genes
        self.fitness = None
        self.age=0
        
    @property
    def genes(self):
        return self._genes
    
    @genes.setter
    def genes(self, new_genes):
        self._genes = new_genes
        self.fitness = None
        self.age=0
        
    def __repr__(self):
        return f"ES Indiv (fitness={self.fitness})" 
        
    def __str__(self):
        return self.__repr__()


class ES:
    def  __init__(self, d, direction="max", n=None, save_path=None, seed=None):
        # Algorithm maximizes, so we multiply fitnesses by -1 if we want to minimize
        if direction == "min":
            self.optim_direction = -1.0
        elif direction == "max":
            self.optim_direction = 1.0
        else:
            raise 
        
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
        self.d = d
            
        if n is None:
            self.n_pop = 4+math.ceil(3*math.log(d))
        else:
            self.n_pop = n
        
        self.gen = 0
        self.hof = [] # Hall of Fame
        self.max_fit=[]
        self.mean_fit=[]
        self.max_age=[]
        self.evaluations=[]
        self.save_path = save_path
        
        self.population = [] 
        
    def __repr__(self):
        return f"ES | d={len(self.population[0].genes)} | n={len(self.population)} | Gen {self.gen}" 
        
    def __str__(self):
        return self.__repr__()
    
    def get_hof(self):
        pop_with_hof = self.population + self.hof
        fitnesses = [- e.fitness for e in pop_with_hof]
        idx = np.argsort(fitnesses)
        self.hof = [copy(pop_with_hof[idx[0]])]
        self.log()
        return self
    
    def log(self):
        #max_fit = self.hof[0].fitness
        max_fit = max(self.population, key=lambda p: p.fitness)
        self.max_fit.append(max_fit)
        self.mean_fit.append(np.mean([i.fitness for i in self.population]))
        evals = 0
        if self.gen > 0:
            evals = self.evaluations[-1]
        evals += len(self.population)
        self.evaluations.append(evals)
        self.max_age.append(np.max([i.age for i in self.population]))

    def export(self):
        return {}

    def load(self, d):
        pass
        
    def plot(self, yscale="linear", data="max", size=8):
        if data == "max":
            x = self.optim_direction * np.array(self.max_fit)
        elif data == "mean":
            x = self.optim_direction * np.array(self.mean_fit)
        elif data == "age":
            x = self.optim_direction * np.array(self.max_age)
        else:
            raise
        plt.figure(figsize=(size*2, size))
        plt.plot(self.evaluations, x, label = data)
        plt.yscale(yscale)
        plt.xlabel(f"Evaluations ({self.gen} gens)")
        plt.ylabel(f"{data} Fitness - {yscale} scale")
        plt.legend()
        plt.title(f"Fitness evolution \n{self}")
        plt.show()
    
    def ask(self):
        self.populate()
        return self.population
    
    def tell(self, pop):
        if self.optim_direction==-1:
            for i in pop:
                i.fitness *= -1.0
        self.population = pop
        self.update()
        self.gen += 1
        for i in self.population:
            i.age +=1
    
    # Specific to algo 
    def populate(self):
        for i in range(self.n_pop):
            new_genes = self.rng.standard_normal(self.d, dtype=np.float32)
            self.population[i].genes = new_genes
        return self
    
    def update(self):
        self.get_hof()
        self.save()