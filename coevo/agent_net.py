  
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import gym

class GlobalAvePool(nn.Module):

    def __init__(self, final_channels):
        super().__init__()
        self._final_channels = final_channels
        self._pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((final_channels, 1, 1)),
            nn.Flatten(),
        )

    def forward(self, input):
        return self._pool(input)

class AgentNet(nn.Module):
    def __init__(self, obs, num_actions):

        super(AgentNet, self).__init__()

        in_channels = obs.shape[0]

        out_channels = 16

        size_pool = 2048
        out_size = size_pool//4

        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(out_channels, out_channels//2, kernel_size=3, stride=1)),
            nn.ReLU(),
            GlobalAvePool(size_pool),
            layer_init(nn.Linear(size_pool, size_pool//2)),
            nn.ReLU(),
            layer_init(nn.Linear(size_pool//2, out_size)),
            nn.ReLU(),
        )

        self.fc_hidden = nn.Linear(in_features=out_size, out_features=128)

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

    def set_parameters(self, param):
        self.parameters = param


def get_state(s, device="cpu"):
    return torch.tensor(s, device=device).unsqueeze(0).float()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Simple function to init layers
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


