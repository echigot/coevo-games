from coevo.env_grid import EnvGrid
from coevo.agent_net import AgentNet
from griddly import GymWrapper
import numpy as np
from coevo import get_state
import copy as cp

# Represents one individual, wether it being an agent
# or an environment
class Individual:

    # For this project, Zelda is the default env
    # We use a personnalized configuration
    default_env = GymWrapper(yaml_file='simple_zelda.yaml', level=0)

    # Maximum number of moves in one game
    nb_steps_max = 100

    # All the allowed movements for one type of game:
    # Labyrinth, Zelda, etc
    # Defined by the define_action_space function
    dic_actions = {}

    def __init__(self, genes):
        self.age = 0
        self.genes = genes

    @property
    def genes(self):
        return self._genes

    # Reset attributes when the genes are (re)defined
    @genes.setter
    def genes(self, new_genes):
        self._genes = new_genes
        self.fitness = 0
        self.age = 0
        self.done = False
        self.last_action = [0,0]

    def __repr__(self):
        return f"ES Indiv (fitness={self.fitness})"

    def __str__(self):
        return self.__repr__()


# Manages an agent as an individual
class AgentInd(Individual):

    def __init__(self, env=None, genes=None, age=1):
        if env is None:
            self.env = Individual.default_env
        else:
            self.env=env
        
        self.total_score = 0
        
        obs = self.env.reset()
        self.agent = AgentNet(get_state(obs), self.env.action_space)

        if genes is None:
            genes = self.agent.get_params()
        else:
            self.agent.set_params(genes)
        
        super(AgentInd, self).__init__(genes)

    # Plays a game between an environment and the agent
    # If envInd is not given, plays on a generic environment
    def play_game(self, envInd=None, render=False):
        if envInd is None:
            play_one_game(self, render=render)
        else:
            play_one_game(self, envInd, render)

    # Computes the indiv fitness according to the score attribute
    def compute_fitness(self):
        self.fitness = self.total_score
    
    # Updates the total score of the agent on
    # all the environments
    def update_scores(self, game_reward):

        # PINSKY fitness, not very successful here
        # fitness = 0
        # if game_reward > 0:
        #     fitness = 1 - self.steps/Individual.nb_steps_max
        # elif game_reward < 0:
        #     fitness = self.steps/Individual.nb_steps_max - 1
        # self.total_score = self.total_score + fitness

        # sum of all the game rewards
        self.total_score = self.total_score + game_reward

# Manages an environment as an individual
class EnvInd(Individual):
    # TODO: make it dynamic
    # Warning: Individual.default_env has a specific size
    height = 13
    width = 9

    def __init__(self, genes=None):
        
        self.env = cp.copy(Individual.default_env)
        self.age = 0

        # Initializes the Cellular Automaton
        self.CA = EnvGrid(width=EnvInd.width, height=EnvInd.height, num_actions=7)
        self.playable = True

        self.min_score = 0
        self.max_score = 0


        if genes is None:
            genes = self.CA.get_params()
        else:
            self.CA.set_params(genes)
        
        super(EnvInd, self).__init__(genes)

    # Generates the level string and plays a game between agent and 
    # related level
    def play_game(self, agent, render=False):
        level_string = self.env_to_string()
        play_one_game(agent, self, level_string=level_string, render=render)

    
    # Transforms a grid of numbers to a string environment, valid
    # as a griddly input to generate a playable map
    def env_to_string(self):
        level_string = ''
        for i in range(EnvInd.width):
            for j in range(EnvInd.height):
                # gets the numeric value of a cell
                block = self.CA.grid[j][i]
                # transforms the value to a character and concatenates it
                level_string+=match_block(block).ljust(4)
            level_string += '\n'
        
        return level_string

    # Computes the indiv fitness according to the score attributes
    def compute_fitness(self):
        # computes the difference between an example map and the indiv map
        
        # obs_th = EnvInd.default_obs.astype(int)
        # obs_gen = self.env.reset(level_string=self.env_to_string()).astype(int)
        # distance = np.sum(np.abs(obs_th - obs_gen), axis=-1)
        # distance = np.sum(distance)
        # self.fitness = distance

        # defines the fitness as the difference between the best game
        # and the worst on the indiv map
        self.fitness = np.abs(self.max_score-self.min_score)

    # Updates the minimum or maximum score done on
    # this specific environment by an agent if necessary
    def update_scores(self, game_reward):
        if game_reward < self.min_score:
            self.min_score = game_reward
        if game_reward > self.max_score:
            self.max_score = game_reward


    def evolve_CA(self):
        self.CA.evolve()

    # Checks if the individual is valid or not and removes it 
    # from the gaming pool if not
    def remove_bad_env(self):
        if (self.CA.is_bad_env()):
            self.playable = False
            self.fitness = -100
            return True
        return False


# Returns the environment encoding for a given CA output value
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
            

# Returns the result of the agent's CNN given a game state
def get_result(agent, obs):
    actions = agent(get_state(obs)).detach().numpy()

    # checks the output encoding
    if (isinstance(agent.n_out, int)):
        a = int(np.argmax(actions))
    else:
        action = int(np.argmax(actions[0][:2]))
        direction = int(np.argmax(actions[0][2:]))
        a = (action, direction)
    return a

# Returns the location of the first occurence of an object 
# on a given environment, None if not present
# object: string of the wanted object's name
def get_object_location(env, object):
    for i in env.get_state()["Objects"]:
        if (i['Name']==object):
            return i['Location']
    
    return None


# Checks if an action is valid before playing
def define_action_space(env):
    env.reset()
    # samples 1000 random actions to fill the dict of playable movements
    for s in range(1000):
        action = tuple(env.action_space.sample())
        if (action in Individual.dic_actions):
            Individual.dic_actions[action] = Individual.dic_actions[action] +1
        else:
            Individual.dic_actions[action] = 1


# Plays a match between two individuals: one agent and one env
# level_string: if a level is generated by the environment
def play_one_game(agentInd=AgentInd(), envInd=EnvInd(), render=False, level_string=None):
    env = envInd.env
    if level_string is None:
        obs = env.reset()
    else:
        obs = env.reset(level_string=level_string)

    game_reward = 0
    done = False
    steps = 0

    # The agent can play until it wins, loses or reaches the maximum 
    # number of moves
    while((not done) and steps < Individual.nb_steps_max):
        if render:
            env.render()
        # asks for the move to play
        result = get_result(agentInd.agent, obs)
        # does the action
        obs, reward, done = do_action(result, env)
        steps = steps + 1
        #update the score of the game
        game_reward = reward + game_reward

    # updates the global score of agent and env
    envInd.update_scores(game_reward)
    agentInd.update_scores(game_reward)
    env.close()

# Does the requested action and returns the result
def do_action(result, env):
    obs_2, reward, done, _ = env.step(result)
    if done:
        obs_2 = env.reset()
    return obs_2, reward, done