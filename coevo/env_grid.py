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

    def __init__(self, width, height, num_actions):

        super(EnvGrid, self).__init__()

        self.height = height
        self.width = width
        
        self.grid = np.zeros((height, width))
        local_grid = np.zeros((3,3))

        self.cell_net = EnvCell(local_grid, num_actions)

        for i in range(width):
            for j in range(height):
                self.grid[j][i] = 0

    def evolve(self):
        old_grid = cp.deepcopy(self.grid)
        if (self.grid.sum() == 0):
            self.grid[self.width//2, self.height//2] = 1
        
        for i in range(self.width):
            for j in range (self.height):
                local_grid = torch.zeros((3,3))
                local_grid = torch.tensor(self.get_local_grid(old_grid, i, j))
                results = self.cell_net(torch.flatten(local_grid).float())
                self.grid[j][i] = np.argmax(results.detach().numpy())

    def is_bad_env(self):
        #more walls than half the surface
        count_wall = np.count_nonzero(self.grid == 6)
        elim = count_wall >= self.height*self.width*3//4
        #more than one agent
        count_agent = np.count_nonzero(self.grid == 1)
        elim = elim or not (0 < count_agent <= 2) 
        #nothing on the map except the agent
        count_objects = np.count_nonzero(self.grid == 4) \
                 + np.count_nonzero(self.grid == 3) \
                 + np.count_nonzero(self.grid  == 5)
        elim = elim or (count_objects <= 0)
        return elim

    def get_local_grid(self, grid, x, y):
        local_grid = np.ones((3,3))*-1
        #position of the submatrix into the new matrix (local_grid)
        x1,x2,y1,y2 = 0,3,0,3
        #coordinates of the submatrix in the original grid
        submatrix_x, submatrix_y, submatrix_w, submatrix_h = x-1, y-1, 3, 3

        if (x == 0):
            x1 = 1
            submatrix_x = 0
            submatrix_w = 2
        if (x == self.width-1):
            x2 = 2
            submatrix_w = 2
        if (y == 0):
            y1 = 1
            submatrix_y = 0
            submatrix_h = 2
        if (y == self.height-1):
            y2 = 2
            submatrix_h = 2

        local_grid[y1:y2, x1:x2] = grid[submatrix_y:submatrix_y+submatrix_h, submatrix_x:submatrix_x+submatrix_w]
        
        return local_grid

    def get_params(self):
        return self.cell_net.get_params()

    def set_params(self, params):
        self.cell_net.set_params(params)




