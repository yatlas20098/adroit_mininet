import os
import subprocess
import gymnasium as gym
from gymnasium import spaces
from torch import nn
import torch
import numpy as np
import threading
import multiprocessing
from helpers import create_mininet_network 
import threading
import queue

class WSNEnvironment(gym.Env):
    metadata = {"render_modes": ["console"]}

    def get_n_envs(self):
        return self._n_clusters

    def get_n_actions(self):
        return self._n_actions
    
    def __init__(self, sim_config, sensor_config, max_steps, device):
        super(WSNEnvironment, self).__init__()
        subprocess.run(["rm", "-rf", sim_config.log_directory])
        os.makedirs(sim_config.log_directory)

        self._cluster_heads = create_mininet_network(sim_config, sensor_config, device)
        self._n_clusters = len(self._cluster_heads)
        self._step_counts = torch.tensor([0] * len(sim_config.clusters), device=device)
        self._n_actions = len(sensor_config.transmission_frequencies)
        
        # Environment parameters
        self._device = device
        self._max_steps = max_steps
        
        ######################### Observation Space #########################
        # The observation space at time t is an (n+2) x n matrix. The ith row
        # in the matrix consists of the recorded temperature (temp_i), 
        # throughput (th_i), transmission rate (tr_i), and n similarity 
        # values (sim_{i,j}) for sensor i. 
        #
        # Temperature:
        #   Unit: Celsuius
        #   Range: -10 to 25
        # Throughput:
        #   Unit: Packets / Second
        #   Range: 0 to +inf
        # Transmission Rate:
        #   Unit: Packets / Second
        #   Range: 0 to +inf
        # Similarity:
        #   Range: 0 or 1 (binary)
        #
        # Note that similarity is computed using the isSimlar function.
        #
        #                       Observation Space Matrix
        # |temp_1| |th_i| |tr_1| |sim_{1,1}| |sim_{1,2}| ... |sim_{1,n}|
        # |temp_2| |th_2| |tr_2} |sim_{2,1}| |sim_{2,2}| ... |sim_{2,n}|
        #                                ... 
        # |temp_n| |th_n| |tr_n| |sim_{n,1}| |sim_{n,2}| ... |sim_{n,n}|
        #####################################################################
        # self.observation_space = spaces.Box(low=-100, high=1000, shape=(self._num_sensors, self._num_sensors + 3,), dtype=np.float32)

        ########################### Action Space ###########################
        # The action space at time t is an n+1 vector. The first n entries
        # correspond to transmisison rates (tr) for sensors. And the n+1th 
        # entry is t' if the enviornment at time t' should be replayed
        # and -1 otherwise. 
        #
        # Transmission Rate:
        #   Unit: Packets / Second
        #   Range: 0 to 250
        #
        # |tr_1| |tr_2| ... |tr_n| |replay|
        ####################################################################
        # self.action_space = spaces.MultiDiscrete(np.array([250] * (self._num_sensors + 1)))

        # Internal state variables
        self.step_count = 0

        # information dictionary
        self.info = {}

        # the generated event
        self.event = None

        # Number of generated events
        self.generated_events = 0
        self.similarity = 0
        self.similarity_penalty = 0

    def _get_observations(self, actions):
        result_queue = queue.Queue()
        threads = []

        for cluster_head, action in zip(self._cluster_heads, actions):
            print("Action: ", action)
            thread = threading.Thread(
                target=lambda h, a: result_queue.put(h.get_observation(a)),
                args=(cluster_head, action)
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        observations = []
        while not result_queue.empty():
            observations.append(result_queue.get())

        return observations

    def reset(self, seed=0):
        # Reset step count
        self._step_counts[0] = 0

        default_actions = []
        for cluster_head in self._cluster_heads:
            n_sensors = cluster_head.get_n_sensors()
            a = [0] * n_sensors
            default_actions.append(a)
            cluster_head.recharge()
        
        observations = self._get_observations(default_actions)

        # state = |similarity, rates, throughput|
        # self._states = (np.column_stack(tuple((obs[0], obs[3], obs[1]))) for obs in observations)
        # self._states = (torch.tensor(state, dtype=torch.float32, device=self._device).squeeze() for state in self._states)
        # self._states = tuple(torch.flatten(state).detach() for state in self._states)
        #_, self._states, _ = zip(*observations)

        _, _, self._states, _ = zip(*observations)

        return self._states, self.info

    def step(self, new_rates_id, actions):
        # Execute one step in the environment
        self._step_counts += 1
        truncated = self._step_counts > self._max_steps
        actions = (action.cpu().detach().numpy() for action in actions)
        
        observations = self._get_observations(actions)
        
        # state = |similarity, rates, throughput|
        #self._states = (np.column_stack(tuple((obs[0], obs[3], obs[1]))) for obs in observations)
        #self._states = (torch.tensor(state, dtype=torch.float32, device=self._device).squeeze() for state in self._states)
        #self._states = tuple((torch.flatten(state) for state in self._states))
        #rewards = tuple((obs[2] for obs in observations))
        awake, terminated, self._states, rewards = zip(*observations)
        terminated = torch.tensor(terminated, device=self._device)
        #self._states = list(zip(connectivity_graphs, redundancy_graphs))

        #print("Sim: ", [obs[0] for obs in observations])
        #print("Rates: ", [obs[3] for obs in observations])
        #print("Throughput: ", [obs[1] for obs in observations])
        print("Rewards: ", rewards)

        return self._states, rewards, awake, terminated, truncated, self.info
