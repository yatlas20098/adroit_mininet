class MininetSimulationConfig:
    def __init__(self):
        # TODO: Fix using multiple clusters
        # A list of 2-element tuples, (ch_id, sensor_ids), where ch_id is the id of a cluster head 
        # and sensor_ids is a list of ids of sensors assigned to that cluster head.
        self.clusters = [(20, list([15,17,18,19,21,22,23,24,25,26]))
                         ]
        # The number of clusters that should be simulated 
        self.num_clusters = 1
        
        # The time in seconds that data should be recorded when gathering an observation
        self.observation_time = 0.04

        # The amount of observations that should be collected before logging data
        self.log_every = 500 
        self.data_points_per_s = 400 
        self.log_directory = f'data/log'
        
        # Window size to use when computing data delivery by node
        self.window_size = 20

class SensorConfig:
    def __init__(self):
        # Size of each transmission in bytes
        self.transmission_size = 1000 
        
        # Possible transmission frequencies (transmissions/s) 
        # self.transmission_frequencies = [25, 50, 100, 200]
        self.transmission_frequencies = [0, 200]

        ############### Energy parameters ############### 
        self.baseline_energy_dist = 10  # Baseline distance
        self.baseline_energy_consumption = 5   # Baseline energy consumption
        self.aE, self.kappaE = 1, 10
        self.sense_energy = 1 * 0.001
        
        self.min_energy = 100.0
        self.base_energy = 10000.0

class MultiAgentDQNConfig:
    def __init__(self):
        self.batch_size = 64 
        self.train_every = 16 
        self.num_episodes = 999999
        self.max_steps = 300
        self.memory_capacity = 2000
        self.lr = 1e-3
        self.gamma = 0.999
        self.eps_start = 1
        self.eps_end = 0.01
        self.eps_decay = 20000
        self.tau = 0.005
