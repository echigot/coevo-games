import numpy as np
import copy as cp
from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import LSTM
import torch.optim
from coevo.env_cell import EnvCell

class EnvGrid():

    def __init__(self, height, width):
        self.height = height
        self.width = width
        
        self.grid = np.zeros(height, width)
        local_grid = np.zeros(3,3)

        self.cell_net = EnvCell(local_grid)

        for i in range(height):
            for j in range(width):
                self.grid[i][j] = 0
        

    def evolve(self):
        old_grid = cp.deepcopy(self.grid)
        for i in range(self.height):
            for j in range (self.width):
                local_grid = old_grid[xxx]
                self.grid[i][j] = self.cell_net(local_grid)

    
    def get_params(self):
        with torch.no_grad():
            params = self.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().numpy()

    def set_params(self, params):
        a = self.parameters()
        torch.nn.utils.vector_to_parameters(torch.tensor(params), a)

        





