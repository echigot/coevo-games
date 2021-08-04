import numpy as np
import copy as cp
from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import LSTM
import torch.optim
from coevo.env_cell import EnvCell

class EnvGrid(nn.Module):

    def __init__(self, height, width, num_actions):

        super(EnvGrid, self).__init__()

        self.height = height
        self.width = width
        
        self.grid = np.zeros((height+1, width+1))
        local_grid = np.zeros(3,3)

        self.cell_net = EnvCell(local_grid)

        for i in range(height):
            for j in range(width):
                self.grid[i][j] = 0


        self.fc_hidden = nn.Linear(in_features=9, out_features=9)
        self.fc_hidden_2 = nn.Linear(in_features=9, out_features=9)

        self.output = nn.Linear(in_features=9, out_features=num_actions)

    
    def forward(self, x):
        x = F.relu(self.fc_hidden(x))
        x = F.relu(self.fc_hidden_2(x))
        return self.output(x)

    def evolve(self):
        old_grid = cp.deepcopy(self.grid)
        if (self.grid.sum() == 0):
            self.grid[self.width//2, self.height//2] = 1
        
        for i in range(self.height):
            for j in range (self.width):
                local_grid = self.get_local_grid(old_grid, i, j)
                self.grid[i][j] = self.cell_net(np.ndarray.flatten(local_grid))


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
        if (x == self.width):
            x2 = 2
        if (y == 0):
            y1 = 1
            submatrix_y = 0
            submatrix_h = 2
        if (y == self.height):
            y2 = 2
        
        local_grid[y1:y2, x1:x2] = grid[submatrix_y:submatrix_y+submatrix_h, submatrix_x:submatrix_x+submatrix_w]
        
        return local_grid


    
    def get_params(self):
        with torch.no_grad():
            params = self.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().numpy()

    def set_params(self, params):
        a = self.parameters()
        torch.nn.utils.vector_to_parameters(torch.tensor(params), a)

        





