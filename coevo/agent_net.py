  
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import gym

class AgentNet(nn.Module):
    def __init__(self, obs, num_actions):

        super(AgentNet, self).__init__()

        in_channels = obs.shape[0]
        width = obs.shape[1]
        height = obs.shape[2]

        out_channels = 16

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
            
        num_linear_units = size_linear_unit(width) * size_linear_unit(height) * out_channels
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=num_actions)
        self.n_out=num_actions

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        # Rectified output from the first conv layer
        x = F.relu(self.conv(x))

        # Rectified output from the final hidden layer
        x = F.relu(self.fc_hidden(x.view(x.size(0),-1)))

        # Returns the output from the fully-connected linear layer
        return self.output(x)

    def get_parameters(self):
        return self.parameters


def get_state(s, device="cpu"):
    return torch.tensor(s, device=device).unsqueeze(0).float()


