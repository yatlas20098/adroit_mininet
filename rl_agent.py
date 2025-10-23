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

from WSN_env import WSNEnvironment
from memory import Transition, PPOMemory
from configs import MininetSimulationConfig, MultiAgentPPOConfig, SensorConfig
from models import CriticNetwork, ActorNetwork 

import time
import os
import pickle
torch.set_num_threads(os.cpu_count())

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def _load_critic(n_obs, critic_lr):
    critic_net = CriticNetwork(n_obs, critic_lr, device).to(device) 

    # Load Critic Network (model and optimizer states)
    checkpoint_critic = torch.load(f'saved_agents/critic_net.pth')
    critic_net.load_state_dict(checkpoint_critic['model_state_dict'])
    critic_net.optimizer.load_state_dict(checkpoint_critic['optimizer_state_dict'])

    return critic_net

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

        self._policy_net = {}
        self._target_net = {}
        self._loss = {}

        load_agent = False 
        n_sensor_features = 6
        if load_agent:
            print("Loading agents")
            self._critic_net = _load_critic(n_sensor_features, config.critic_lr)
            self._actor_net = _load_actor(self._n_actions, n_sensor_features, config.actor_lr)
        else:
            # Iniatlize Nerual Networks
            self._critic_net = CriticNetwork(n_sensor_features, config.critic_lr, device).to(device)
            self._actor_net = ActorNetwork(self._n_actions, n_sensor_features, config.actor_lr, device).to(device)
        
        # Replay memory for trainning
        self._memory = [PPOMemory(batch_size=self._config.batch_size) for _ in range(self._n_envs)]

        self._steps_done = 0
        self._episode_durations = []
        self._energy_consumption = []

        self._state, self._info = self._env.reset()

        self._critic_loss_log = []
        self._actor_loss_log = []
        self._rewards_log = []
        self._episode_lengths_log = []

    def _save_agent(self):
        # Save the actor network
        torch.save({
            'model_state_dict': self._actor_net.state_dict(),
            'optimizer_state_dict': self._actor_net.optimizer.state_dict(),
        }, f'saved_agents/actor_net.pth')

    def _save_critic(self):
        # Save the critic network
        torch.save({
            'model_state_dict': self._critic_net.state_dict(),
            'optimizer_state_dict': self._critic_net.optimizer.state_dict(),
        }, f'saved_agents/critic_net.pth')

    def _optimize_model(self):
        critic_total_loss = 0
        actor_total_loss = 0

        for epoch in range(self._config.n_epochs):
            critic_epoch_total_loss = 0
            actor_epoch_total_loss = 0

            for env in range(self._n_envs):
                transitions, batches = self._memory[env].generate_batches()
                transitions = Transition(*zip(*transitions))

                states_arr = transitions.state
                actions_arr = torch.stack(transitions.action).squeeze()
                values_arr = torch.stack(transitions.value).squeeze()
                old_probs_arr = torch.stack(transitions.probs).squeeze()
                awake_arr = torch.stack(transitions.awake).squeeze()
                truncated_arr = torch.tensor(transitions.truncated, device=device).squeeze()
                terminated_arr = torch.tensor(transitions.truncated, device=device).squeeze()
                done_arr = truncated_arr | terminated_arr

                rewards_arr = torch.stack(transitions.reward).squeeze()

                if(epoch == 0):
                    self._rewards_log.append(rewards_arr.view(-1).sum().detach().cpu().numpy())

                for batch in batches:
                    states = [states_arr[i] for i in batch]
                    states = Batch.from_data_list(states)
                    last_state = states_arr[batch[-1]]

                    actions = actions_arr[batch] # Batch Size X N_Nodes
                    values = values_arr[batch] # Batch Size X N_Nodes
                    old_probs = old_probs_arr[batch] # Batch Size * N_Nodes
                    truncs = truncated_arr[batch].float().unsqueeze(1)
                    terms = terminated_arr[batch].float().unsqueeze(1)
                    dones = done_arr[batch].float().unsqueeze(1) # Batch Size, 1
                    awake = awake_arr[batch] # Batch Size X N_Nodes
                    reward = rewards_arr[batch] # Batch Size X N_Nodes

                    with torch.no_grad():
                        last_value = self._critic_net(last_state).squeeze()
                        values = torch.cat([values, last_value.unsqueeze(0)], dim=0)
                        deltas = reward + self._config.gamma*values[1:]*(1.0 - terms) - values[:-1]
                        advantage = torch.zeros_like(deltas)

                    gae = 0.0
                    for t in reversed(range(len(deltas))):
                        mask = 1.0 - truncs[t]
                        gae = (deltas[t] + self._config.gamma*self._config.gae_lambda*gae) * mask
                        advantage[t] = gae
                
                    norm_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                    dists = self._actor_net(states)
                    critic_values = self._critic_net(states).squeeze().reshape(self._config.batch_size, -1)
                    
                    new_probs = dists.log_prob(actions.view(-1)).reshape(self._config.batch_size, -1)
                    prob_ratio = torch.exp(new_probs - old_probs).reshape(self._config.batch_size, -1)

                    weighted_probs = norm_advantage * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 
                                                            (1.0 - self._config.policy_clip),
                                                            (1.0 + self._config.policy_clip))*norm_advantage

                    # Note networks are only trained using awake agents 
                    not_done_mask = (dones == 0).squeeze()
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
                    actor_loss = actor_loss[not_done_mask]
                    awake = awake[not_done_mask]

                    #actor_loss = actor_loss.view(-1)[awake.view(-1)]
                    n_agents = actor_loss.shape[1]
                    entropy = dists.entropy().mean()
                    actor_loss_sum = 0
                    for i in range(n_agents):
                        single_actor_loss = actor_loss[:, i][awake[:,i]]
                        if single_actor_loss.numel() == 0:
                            continue
                        
                        single_actor_loss = single_actor_loss.mean()
                        actor_loss_sum += single_actor_loss
                    
                    self._actor_net.optimizer.zero_grad()
                    (actor_loss_sum - self._config.entropy_coef*entropy).backward()

                    torch.nn.utils.clip_grad_norm_(self._actor_net.parameters(), max_norm=5.0)
                    self._actor_net.optimizer.step()
                    actor_loss = actor_loss_sum
                     
                    returns = (advantage + values[:-1]).detach()
                    critic_loss_sum = 0
                    for i in range(n_agents):
                        final_mask = not_done_mask & awake[:, i]
                        single_critic_loss = (returns[final_mask, i] - critic_values[final_mask, i])**2
                        if single_critic_loss.numel() == 0:
                            continue

                        critic_loss_sum += single_critic_loss.mean()
                    
                    critic_loss = critic_loss_sum
                    # critic_loss = critic_loss.mean(dim=1)

                    #critic_loss = critic_loss[not_done_mask].mean()

                    self._critic_net.optimizer.zero_grad()
                    (self._config.value_loss_coef*critic_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self._critic_net.parameters(), max_norm=5.0)
                    self._critic_net.optimizer.step()
                    
                    actor_epoch_total_loss += actor_loss
                    critic_epoch_total_loss += critic_loss

                    total_loss = actor_loss + critic_loss
                
                batch_len = self._config.train_every // self._config.batch_size
                print(f"Env {env} Epoch {epoch}")
                print("\tCritic loss: ", critic_epoch_total_loss / batch_len)
                print("\tActor loss: ", actor_epoch_total_loss / batch_len) 
                critic_total_loss += critic_epoch_total_loss / batch_len
                actor_total_loss += actor_epoch_total_loss / batch_len
                print("")

        # Save the network
        self._save_agent()
        self._save_critic()

        self._actor_loss_log.append((actor_total_loss / self._config.n_epochs).detach().cpu().numpy())
        self._critic_loss_log.append((critic_total_loss / self._config.n_epochs).detach().cpu().numpy())

        for env in range(self._n_envs):
            self._memory[env].clear()


    """
    def _optimize_model(self):
        self._optimize_model_for_cluster(0)
        self._memory[0].clear()
    """

    def _select_action(self):
        actions, values, probs = [], [], []

        for i in range(self._sim_config.num_clusters):
            dists = self._actor_net(self._state[i])
            value = self._critic_net(self._state[i]).detach()
            action = dists.sample()
            prob = dists.log_prob(action).detach()

            actions.append(action)
            values.append(value)
            probs.append(prob)
        return actions, probs, values
    
    def train(self):
        train_steps = -1
        throughput_reward_log = [-1 for _ in range(10)]
        init_actor_lr = self._config.actor_lr         
        init_critic_lr = self._config.critic_lr
        init_entropy_coef = self._config.entropy_coef
        first_step = True
        throughput_rewards, similarity_rewards = None, None

        for i_episode in range(self._config.num_episodes):
            # Initialize the environment and get its state
            self._state, self._info = self._env.reset()
            print(self._state)

            # for t in count():
            for t in range(self._config.max_steps):
                print(f'\n\nEpisode {i_episode}, Step: {t}, Buffer size: {len(self._memory[0])}/{self._config.train_every}')

                actions, probs, values = self._select_action()
                
                # Sample the next frame from the enviornment, and receive a reward
                self._next_state, rewards, awake, terminated, truncated, _ = self._env.step(self._steps_done, actions)

                #prev_throughput_rewards, prev_similarity_rewards = throughput_rewards, similarity_rewards

                # Move the reward onto the correct device (memory, cpu, or gpu)
                #throughput_rewards = [torch.tensor(reward["throughput"], device=device, dtype=torch.float32) for reward in rewards]
                #similarity_rewards = [torch.tensor(reward["similarity"], device=device, dtype=torch.float32) for reward in rewards]
                
                dones = torch.logical_or(terminated, truncated)
                for cluster_idx in range(self._sim_config.num_clusters):
                    done = dones[cluster_idx].item()

                    if not first_step:
                        # Store the transition in memory
                        self._memory[cluster_idx].push(self._state[cluster_idx], terminated[cluster_idx], truncated[cluster_idx], awake[cluster_idx], actions[cluster_idx], probs[cluster_idx], values[cluster_idx].detach(), self._next_state[cluster_idx], rewards[cluster_idx].detach().to(device))

                # Move to the next state
                self._state = self._next_state
                first_step = False 
                
                train_steps += self._sim_config.num_clusters
                if len(self._memory[0]) >= self._config.train_every:
                    #train_steps = 0

                    # Perform one step of the optimization (on the policy network)
                    self._optimize_model()
                    
                    with open(f'rl_agent_figure_data.pkl', 'wb') as file:
                        pickle.dump((self._critic_loss_log, self._actor_loss_log, self._rewards_log, self._episode_lengths_log), file)
                
                # Wait for all envs to termiante before reset 
                done_idxs = torch.argwhere(dones)
                if done_idxs.numel() == self._sim_config.num_clusters:
                    print("Done")
                    self._episode_lengths_log.append(t + 1)
                    break
                
if __name__ == '__main__':
    training_config = MultiAgentPPOConfig()
    sim_config = MininetSimulationConfig()
    sensor_config = SensorConfig() 

    agent = WSN_agent(sim_config, sensor_config, training_config)
    agent.train()
