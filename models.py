import gymnasium as gym
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import struct
import time
import threading
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv


from torch.distributions.categorical import Categorical 

import os

from WSN_env import WSNEnvironment

class DDQN(nn.Module):
    def __init__(self, n_actions, n_observations, lr, device):
        super(DDQN, self).__init__()
        self._device = device

        w = 128 # number of nodes in a hidden layer
        num_hidden_layers = 12  

        layers = [nn.Linear(n_observations, w), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(w,w))
            layers.append(nn.ReLU())

        self._layers = nn.Sequential(*layers)

        # Initalize random weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)

        self._fc1 = nn.Linear(w, w)
        std = math.sqrt(2.0 / (64))
        nn.init.normal_(self._fc1.weight, mean=0.0, std=std)                                                 
        self._fc1.bias.data.fill_(0.0)
        self._V = nn.Linear(w, 1)
        self._A = nn.Linear(w, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)


     # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, state):
        x = self._layers(state).to(self._device)
        x = F.relu(self._fc1(x))

        V = self._V(x)
        A = self._A(x)
        Q = V + (A - A.mean(dim=0, keepdim=True))

        return Q

# (s,a) -> Q
class DQN(nn.Module):
    def __init__(self, n_actions, n_observations, num_sensors, lr, device):
        super(DQN, self).__init__()
        self._device = device

        w = 512 # number of nodes in a hidden layer
        num_hidden_layers = 64 

        layers = [nn.Linear(n_observations, w), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(w,w))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(w, n_actions))
        self._layers = nn.Sequential(*layers)

        # Initalize random weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)


    # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, state):
        print(state.x)
        x = state.x.float()
        return self._layers(x).to(self._device)
