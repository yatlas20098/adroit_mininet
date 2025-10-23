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
from torch_geometric.nn import GATv2Conv


from torch.distributions.categorical import Categorical 

import os

from WSN_env import WSNEnvironment

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, edge_dim=1):
        super().__init__()
        self.device = device
        self.gat1 = GATv2Conv(in_channels=input_dim, out_channels=hidden_dim, heads=4, concat=True, dropout=0.0, edge_dim=edge_dim)

        self.to(device)

    def forward(self, x, edge_attr, edge_index):
        x = F.relu(self.gat1(x.float(), edge_index, edge_attr))
        
        return x

# (s) -> a
class ActorNetwork(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def __init__(self, n_actions, n_node_features, lr, device, chkpt_dir=r".\actor.chkpt", hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.device = device
        self._checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self._graph_encoder = GraphEncoder(n_node_features, hidden_dim, device)

        self._mlp = nn.Sequential(
            nn.Linear(4*hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self._initialize_weights()
        self.to(device)
 
    # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, state):
        graph_embeddings = self._graph_encoder(state.x, state.edge_attr, state.edge_index)
        logits = self._mlp(graph_embeddings)

        return Categorical(logits=logits) # switch to categorical distrbuition

    def save_checkpoint(self):
        torch.save(self.state_dict(), self._checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self._checkpoint_file))

# (s, a) -> Q
class CriticNetwork(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight)

                if m.bias is not None:
                    init.zeros_(m.bias)


    def __init__(self, input_dim, lr, device, chkpt_dir=r".\critic.chkpt", hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.device = device

        self._checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self._graph_encoder = GraphEncoder(input_dim, hidden_dim, device)

        self._mlp = nn.Sequential(
            nn.Linear(4*hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self._initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, state):
        graph_embeddings = self._graph_encoder(state.x, state.edge_attr, state.edge_index)
        values = self._mlp(graph_embeddings) 

        return values

    def save_checkpoint(self):
        torch.save(self.state_dict(), self._checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self._checkpoint_file))
