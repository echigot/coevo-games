from coevo.individual import EnvInd
import gym
import pytest

from griddly import GymWrapperFactory, gd
from griddly.util.environment_generator_generator import EnvironmentGeneratorGenerator

import matplotlib.pyplot as plt
from coevo import AgentInd




def test_gen_map():

    env_ind = EnvInd()