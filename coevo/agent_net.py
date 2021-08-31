  
from gym.spaces.discrete import Discrete
from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import LSTM
import torch.optim
import numpy as np
from gym import spaces

# Manages the agent's CNN
class AgentNet(nn.Module):
    def __init__(self, obs, action_space):

        super(AgentNet, self).__init__()

        in_channels = obs.shape[1]
        
        # allows more flexibility regarding the game
        if (isinstance(action_space, int)):
            # Labyrinth
            self.n_out = action_space
        elif (isinstance(action_space, Discrete)):
            # Zelda
            self.n_out = action_space.n
        else:
            self.n_out=np.sum(action_space.nvec[:])

        kernel_size = 3
        linear_flatten = np.prod(obs.shape[1:])*kernel_size*4

        # convolutionnal layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size, padding=1),
            nn.Identity(),
            nn.Conv2d(6, 12, kernel_size, padding=1),
            nn.Identity(),
            nn.Flatten(),
        )

        # recurrent layer
        hidden_state = torch.rand(1, 1, 128)
        cell_state = torch.rand(1, 1, 128)

        self.fc_hidden = nn.LSTM(linear_flatten, 128)
        self.hidden = (hidden_state, cell_state)
        
        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=self.n_out)
        


    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        # Rectified output from the first conv layer
        x = self.conv(x)
        
        # Rectified output from the final hidden layer
        x = x.unsqueeze(0)
        y, self.hidden = self.fc_hidden(x, self.hidden)
        # Returns the output from the fully-connected linear layer

        y = y.contiguous().view(-1, 128)

        y = self.output(y)

        return y

    def get_params(self):
        with torch.no_grad():
            params = self.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().numpy()

    def set_params(self, params):
        a = self.parameters()
        torch.nn.utils.vector_to_parameters(torch.tensor(params), a)

# Converts from binary 3D matrix to [0-6] 2D matrix
# Also, converts said matrix to the right format
def get_state(s, device="cpu"):
    frame = torch.zeros(s.shape[1], s.shape[2])
    for i in range (s.shape[0]) :
        frame = frame + s[i]*(i+1)

    return frame.unsqueeze(0).unsqueeze(0).float()
