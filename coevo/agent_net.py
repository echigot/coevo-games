  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import LSTM
import torch.optim
import numpy as np
import gym


class AgentNet(nn.Module):
    def __init__(self, obs, num_actions):

        super(AgentNet, self).__init__()

        in_channels = obs.shape[1]

        kernel_size = 3
        linear_flatten = np.prod(obs.shape[1:])*kernel_size

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size, padding=1),
            nn.Identity(),
            nn.Conv2d(6, 12, kernel_size, padding=1),
            nn.Identity(),
            nn.Flatten(),
        )

        #couche récurrente
        self.fc_hidden = nn.LSTM(linear_flatten, 128)

        hidden_state = torch.rand(1, 1, 128)
        cell_state = torch.rand(1, 1, 128)
        self.hidden = (hidden_state, cell_state)
        #self.fc_hidden = nn.Linear(in_features=linear_flatten, out_features=128)
        
        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=num_actions)
        self.n_out=num_actions


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
        x, self.hidden = self.fc_hidden(x, self.hidden)
        #print("third ", x)
        # Returns the output from the fully-connected linear layer

        x = self.output(x)
        #print("last ", x)

        #print("params = ",self.get_params())
        return x

    def get_params(self):
        with torch.no_grad():
            params = self.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().numpy()

    def set_params(self, params):
        a = torch.tensor(params).float()
        torch.nn.utils.vector_to_parameters(a, self.parameters())


def get_state(s, device="cpu"):
    return torch.tensor(s, device=device).unsqueeze(0).float()
