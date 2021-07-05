from coevo.individual import Individual, get_object_location

class EvoStrat:

    def evolve(elites, size_pop):

        return elites

    def select(population, nb_elites):
        temp_pop = population.copy()
        temp_pop.sort(key = lambda x : x.indiv.fitness, reverse = True)
        elites = temp_pop[0:nb_elites]

        for i in elites:
            print()
            print('distance = ', i.indiv.fitness)
            print('position = ', get_object_location(i.env, 'avatar'))
            print('goal = ', get_object_location(i.env, 'exit'))

        return elites


class Element:

    def __init__(self, agent, env, indiv) :
        self.agent = agent
        self.env = env
        self.indiv = indiv