from .evolution_strategies import *


class Canonical(ES):
    """Canonical ES for optimisation of float arrays"""
    def __init__(self, d, direction="max", n=None, n_parents=None, save_path=None):
        super().__init__(d, direction, n, save_path)
        
        # Number of parents selected
        if n_parents is None:
            n_parents = int(self.n_pop/4)
        self.n_parents = n_parents

        assert(self.n_parents <= self.n_pop)

        self.sigma = 1
        self.lr = 1
        self.c_sigma_factor = 1

        # Current solution (The one that we report).
        self.mu = np.random.rand(self.d)
        # Computed update, step in parameter space computed in each iteration.
        self.step = 0

        # Compute weights for weighted mean of the top self.n_parents offsprings
        # (parents for the next generation).
        self.w = np.array([np.log(self.n_parents + 0.5) - np.log(i) for i in range(1, self.n_parents + 1)])
        self.w /= np.sum(self.w)

        # Noise adaptation stuff.
        self.p_sigma = np.zeros(self.d)
        self.u_w = 1 / float(np.sum(np.square(self.w)))
        self.c_sigma = (self.u_w + 2) / (self.d + self.u_w + 5)
        self.c_sigma *= self.c_sigma_factor
        self.const_1 = np.sqrt(self.u_w * self.c_sigma * (2 - self.c_sigma))

        self.s = np.random.randn(self.d, self.n_pop)

        
    def __repr__(self):
        return f"Canonical | d={len(self.population[0].genes)} | ({len(self.population)}, {self.n_parents}) | Gen {self.gen}" 
    
    def populate(self):
        for i in range(self.n_pop):
            new_genes = self.mu + self.sigma * self.s[:, i]
            self.population[i].genes = new_genes 
        return self

    def back_random(self, genes_after):
        return (genes_after - self.mu)/self.sigma    
    
    def update(self):
        d =len(self.population[0].genes)
        n = self.n_pop
        
        fitnesses = [- i.fitness for i in self.population]
        idx = np.argsort(fitnesses) # indices from highest fitness to lowest

        step = np.zeros(self.d)
        
        # print([i.fitness for i in self.population])
        # print([self.population[i].fitness for i in idx])
        for i in range(self.n_parents):
            ind = self.population[idx[i]]
            vect = self.back_random(ind.genes)
            step += self.w[i] * vect
                
        self.step = self.lr * self.sigma * step
        self.mu += self.step
        
        # Noise adaptation stuff.
        # self.p_sigma = (1 - self.c_sigma) * self.p_sigma + self.const_1 * step
        # self.sigma = self.sigma * np.exp((self.c_sigma / 2) * (np.sum(np.square(self.p_sigma)) / self.d - 1))
        
        self.s = np.random.randn(d, n)

        self.get_hof()

    def export(self):
        return {
            "n_parents":list(self.state.n_parents),
            "sigma":list(self.state.sigma)
        }
        
    def load(self, d):
        self.state.n_parents = d["n_parents"]
        self.state.sigma = d["sigma"]
