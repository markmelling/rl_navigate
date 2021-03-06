import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np
import random
from collections import namedtuple, deque


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2_val = nn.Linear(fc1_units, fc2_units)
        self.fc2_adv = nn.Linear(fc1_units, fc2_units)

        self.fc3_val = nn.Linear(fc2_units, action_size)
        self.fc3_adv = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        val = F.relu(self.fc2_val(x))
        adv = F.relu(self.fc2_adv(x))
        val = self.fc3_val(val).expand(x.size(0), self.action_size)
        adv = self.fc3_adv(adv)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        return x

    
class ConvQNetwork(nn.Module):
    """Actor (Policy) Model."""

    # in_channels is the number of features:
    # for a grey scale image this is 1
    # for a colour image it's 3
    def __init__(self, state_size, action_size, seed, in_channels=3, fc_units=[64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(ConvQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        # same convolution layers as original paper
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        in_features = 7*7*64
        
        prev_n_units = in_features
        self.fc_layers = []
        for n_units in fc_units:
            self.fc_layers.append(nn.Linear(prev_n_units, n_units))
            prev_n_units = n_units
        # final output layer
        self.output_layer = nn.Linear(prev_n_units, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        #print('forward state.size', state.size())
        #x = state.view(state.size(1), -1)

        #x = F.relu(self.conv1(torch.transpose(state, 0, 1)))
        x = state.transpose(1,3).float()
        #print('x.size', x.size())

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        #print('size of x', x.size(0))
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        x = self.output_layer(x)
        return x
    
class DuelingConvQNetwork(nn.Module):
    """Actor (Policy) Model."""

    # in_channels is the number of features:
    # for a grey scale image this is 1
    # for a colour image it's 3
    def __init__(self, state_size, action_size, seed, in_channels=3, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingConvQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        in_features = 7*7*64

        self.fc1_val = nn.Linear(in_features, fc1_units)
        self.fc1_adv = nn.Linear(in_features, fc1_units)
        
        self.fc2_val = nn.Linear(fc1_units, fc2_units)
        self.fc2_adv = nn.Linear(fc1_units, fc2_units)

        self.fc3_val = nn.Linear(fc2_units, action_size)
        self.fc3_adv = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        #print('forward state.size', state.size())
        #x = state.view(state.size(1), -1)

        #x = F.relu(self.conv1(torch.transpose(state, 0, 1)))
        x = state.transpose(1,3).float()
        #print('x.size', x.size())
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        val = F.relu(self.fc1_val(x))
        adv = F.relu(self.fc1_adv(x))
        val = F.relu(self.fc2_val(val))
        adv = F.relu(self.fc2_val(adv))
        val = self.fc3_val(val).expand(x.size(0), self.action_size)
        adv = self.fc3_adv(adv)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        return x

