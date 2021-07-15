  
from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import LSTM
import torch.optim
import numpy as np
import gym


class AgentNet(nn.Module):
    def __init__(self, obs, action_space):

        super(AgentNet, self).__init__()

        in_channels = obs.shape[1]
        self.n_out=np.sum(action_space.nvec[:])

        kernel_size = 3
        linear_flatten = np.prod(obs.shape[1:])*kernel_size*4

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size, padding=1),
            nn.Identity(),
            nn.Conv2d(6, 12, kernel_size, padding=1),
            nn.Identity(),
            nn.Flatten(),
        )

        #couche récurrente
        hidden_state = torch.rand(1, 1, 128)
        cell_state = torch.rand(1, 1, 128)

        self.fc_hidden = nn.LSTM(linear_flatten, 128)
        self.hidden = (hidden_state, cell_state)

        self.fc_hidden_sc = nn.LSTM(128, 128)
        self.hidden_sc = (hidden_state, cell_state)

        #self.fc_hidden_th = nn.LSTM(128, 128)
        #self.hidden_th = (hidden_state, cell_state)
        #self.fc_hidden = nn.Linear(in_features=linear_flatten, out_features=128)
        
        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=self.n_out)
        


    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        #print("first", x)
        # Rectified output from the first conv layer
        x = self.conv(x)
        #print("sec ", x)
        
        # Rectified output from the final hidden layer
        #x = F.relu(self.fc_hidden(x))#, self.hx))
        x = x.unsqueeze(0)
        #hx = torch.zeros(1, 1, 128)
        y, self.hidden = self.fc_hidden(x, self.hidden)

        #y, self.hidden_sc = self.fc_hidden_sc(y, self.hidden_sc)

        #y, self.hidden_th = self.fc_hidden_th(y, self.hidden_th)
        #print("third ", x)
        # Returns the output from the fully-connected linear layer

        y = y.contiguous().view(-1, 128)

        y = self.output(y)
        #print("last ", x)

        #print("params = ",self.get_params())
        return y

    def get_params(self):
        with torch.no_grad():
            params = self.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().numpy()

    def set_params(self, params):
        a = torch.tensor(params).float()
        torch.nn.utils.vector_to_parameters(a, self.parameters())


def get_state(s, device="cpu"):
    frame = torch.zeros(s.shape[1], s.shape[2])
    for i in range (s.shape[0]) :
        frame = frame + s[i]*(i+1)

    return frame.unsqueeze(0).unsqueeze(0).float()
