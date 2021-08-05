
import numpy as np

import numpy as np
import copy as cp
from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import LSTM
import torch.optim

class EnvCell(nn.Module):

      def __init__(self, local_grid, num_actions):

         super(EnvCell, self).__init__()

         self.local_grid = local_grid

         self.fc_hidden = nn.Linear(in_features=9, out_features=9)
         self.fc_hidden_2 = nn.Linear(in_features=9, out_features=9)

         self.output = nn.Linear(in_features=9, out_features=num_actions)


      def forward(self, x):
         x = F.relu(self.fc_hidden(x))
         x = F.relu(self.fc_hidden_2(x))
         return self.output(x)

      def get_params(self):
        with torch.no_grad():
            params = self.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().numpy()

      def set_params(self, params):
         a = self.parameters()
         torch.nn.utils.vector_to_parameters(torch.tensor(params), a)