from itertools import count
from torch_geometric.loader import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
from torch.distributions.categorical import Categorical 
from torch_geometric.data import Batch
import math
import numpy as np

from WSN_env import WSNEnvironment
from memory import Transition, ReplayMemory 
from configs import MininetSimulationConfig, MultiAgentDQNConfig, SensorConfig
from models import DDQN,DQN

import time
import os
import pickle
torch.set_num_threads(os.cpu_count())

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def _load_actor(sampling_freq, n_obs, actor_lr):
    actor_net = ActorNetwork(sampling_freq, n_obs, actor_lr, device).to(device)

    # Load Actor Network (model and optimizer states)
    checkpoint_actor = torch.load(f'saved_agents/actor_net.pth')
    actor_net.load_state_dict(checkpoint_actor['model_state_dict'])
    actor_net.optimizer.load_state_dict(checkpoint_actor['optimizer_state_dict'])
   
    return actor_net

class WSN_agent:
    def __init__(self, mininet_simulation_config, sensor_config, config):
        self._config = config 
        self._sim_config = mininet_simulation_config
        self._env = WSNEnvironment(mininet_simulation_config, sensor_config, self._config.max_steps, device) 

        self._n_actions = self._env.get_n_actions()
        self._n_envs = self._env.get_n_envs()

        self._loss = {}

        load_agent = False 
        n_sensor_features = 1

        if load_agent:
            self._load_model(n_sensor_features, actor_lr)
        else:
            # Iniatlize Nerual Networks
            self._policy_net = DDQN(self._n_actions, n_sensor_features, self._config.lr, device).to(device)
            self._target_net = DDQN(self._n_actions, n_sensor_features, self._config.lr, device).to(device)
        
        # Replay memory for trainning
        self._memory = [ReplayMemory(capacity=self._config.memory_capacity) for _ in range(self._n_envs)]

        self._steps_done = 0
        self._episode_durations = []
        self._energy_consumption = []

        self._state, self._info = self._env.reset()

        self._loss_log = []
        self._rewards_log = []
        self._episode_lengths_log = []

    def _save_model(self):
        torch.save({
            'model_state_dict': self._policy_net.state_dict(),
            'optimizer_state_dict': self._policy_net.optimizer.state_dict(),
        }, f'saved_agents/policy_dqn_net.pth')

        torch.save({
            'model_state_dict': self._target_net.state_dict(),
            'optimizer_state_dict': self._target_net.optimizer.state_dict(),
        }, f'saved_agents/target_dqn_net.pth')


    def _load_model(self, n_sensor_features, actor_lr):
        self._policy_net = DDQN(self._n_actions, n_sensor_features, self._config.lr, device).to(device)
        self._target_net = DDQN(self._n_actions, n_sensor_features, self._config.lr, device).to(device)

        # Load Actor Network (model and optimizer states)
        checkpoint_policy = torch.load(f'saved_agents/policy_dqn_net.pth')
        self._policy_net.load_state_dict(checkpoint_actor['model_state_dict'])
        self._policy_net.optimizer.load_state_dict(checkpoint_actor['optimizer_state_dict'])

        check_point_target = torch.load(f'saved_agents/target_dqn_net.pth')
        self._target_net.load_state_dict(checkpoint_actor['model_state_dict'])
        self._target_net.optimizer.load_state_dict(checkpoint_actor['optimizer_state_dict'])

    def _optimize_model(self):
        total_loss = 0

        for env in range(self._n_envs):
            if(len(self._memory[env]) < self._config.batch_size):
                return 

            batch = self._memory[env].sample(self._config.batch_size)
            transitions = Transition(*zip(*batch))

            states = torch.stack(transitions.state)
            next_states = torch.stack(transitions.next_state)
            actions = torch.stack(transitions.action).squeeze()
            awake = torch.stack(transitions.awake).squeeze()
            truncated = torch.tensor(transitions.truncated, device=device).squeeze()
            terminated = torch.tensor(transitions.truncated, device=device).squeeze()
            dones = (truncated | terminated).float()
            rewards = torch.stack(transitions.reward).squeeze()

            not_done_mask = (dones == 0).squeeze()
            states = states[not_done_mask]

            next_states = next_states[not_done_mask]
            
            n_sensors = self._env.get_n_sensors(env)
            state_action_values = self._policy_net(states).gather(2, actions.unsqueeze(2)).squeeze()
            next_state_values = torch.zeros(self._config.batch_size, n_sensors, device=device)
            with torch.no_grad():
                max_values = self._target_net(next_states).max(2)[0]
            next_state_values[not_done_mask] = max_values
            expected_state_action_values = (next_state_values * self._config.gamma) + rewards
            expected_state_action_values = expected_state_action_values.squeeze()

            awake = awake[not_done_mask]
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values[awake], expected_state_action_values[awake])
            total_loss += loss

            self._policy_net.optimizer.zero_grad()
            loss.backward()
            self._policy_net.optimizer.step()

        
        print(f'Total loss: {total_loss}')
        self._loss_log.append(total_loss)

    def _select_action(self):
        actions = []
        eps_threshold = self._config.eps_end + (self._config.eps_start - self._config.eps_end) * math.exp(-1. * self._steps_done / self._config.eps_decay)

        for i in range(self._sim_config.num_clusters):
            n_sensors = self._env.get_n_sensors(i)
            action = torch.randint(0, self._env.get_n_actions(), (n_sensors,), device=device)

            for j in range(n_sensors):
                if np.random.rand() > eps_threshold:
                    values = self._policy_net(self._state[i][j])
                    action[j] = torch.argmax(values)

            actions.append(action)

        return actions
    
    def train(self):
        self._steps_done = 0

        for i_episode in range(self._config.num_episodes):
            # Initialize the environment and get its state
            self._state, self._info = self._env.reset()
            train_steps = 0

            for t in range(self._config.max_steps):
                print(f'\n\nEpisode {i_episode}, Step: {t}, Train Counter: {train_steps}/{self._config.train_every}')

                actions = self._select_action()
                
                # Sample the next frame from the enviornment, and receive a reward
                self._next_state, rewards, awake, terminated, truncated, _ = self._env.step(self._steps_done, actions)

                dones = torch.logical_or(terminated, truncated)
                for cluster_idx in range(self._sim_config.num_clusters):
                    done = dones[cluster_idx].item()

                    # Store the transition in memory
                    self._memory[cluster_idx].push(self._state[cluster_idx], terminated[cluster_idx], truncated[cluster_idx], awake[cluster_idx], actions[cluster_idx], self._next_state[cluster_idx], rewards[cluster_idx].detach().to(device))

                # Move to the next state
                self._state = self._next_state
                
                train_steps += self._sim_config.num_clusters
                self._steps_done += self._sim_config.num_clusters
                if train_steps >= self._config.train_every:
                    train_steps = 0

                    # Perform one step of the optimization (on the policy network)
                    self._optimize_model()

                    target_net_state_dict = self._target_net.state_dict()
                    policy_net_state_dict = self._policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self._config.tau + target_net_state_dict[key]*(1-self._config.tau)
                    self._target_net.load_state_dict(target_net_state_dict)
                    
                    with open(f'rl_agent_figure_data.pkl', 'wb') as file:
                        pickle.dump((self._loss_log, self._rewards_log, self._episode_lengths_log), file)
                
                # Wait for all envs to termiante before reset 
                done_idxs = torch.argwhere(dones)
                if done_idxs.numel() == self._sim_config.num_clusters:
                    self._episode_lengths_log.append(t + 1)
                    break
                
if __name__ == '__main__':
    training_config = MultiAgentDQNConfig()
    sim_config = MininetSimulationConfig()
    sensor_config = SensorConfig() 

    agent = WSN_agent(sim_config, sensor_config, training_config)
    agent.train()
