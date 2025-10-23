class MininetSimulationConfig:
    def __init__(self):
        # TODO: Fix using multiple clusters
        # A list of 2-element tuples, (ch_id, sensor_ids), where ch_id is the id of a cluster head 
        # and sensor_ids is a list of ids of sensors assigned to that cluster head.
        """
        self.clusters = [(19, list(range(13,16)) + [17,18] + list(range(21, 26))),
                         (29, list(range(26, 29)) + list(range(30, 36)) + [37]) 
                         ]
        """

        self.clusters = [(20, list([15,17,18,19,21,22,23,24,25,26]))
                         ]

        # The time in seconds that data should be recorded when gathering an observation
        self.observation_time = 0.04

        # The number of clusters that should be simulated 
        self.num_clusters = 1
        
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
        self.transmission_frequencies = [25, 50, 100, 200]

        ############### Energy parameters ############### 
        self.baseline_energy_dist = 10  # Baseline distance
        self.baseline_energy_consumption = 5   # Baseline energy consumption
        self.aE, self.kappaE = 1, 10
        self.sense_energy = 1 * 0.001
        
        self.min_energy = 100.0
        self.base_energy = 10000.0

class MultiAgentPPOConfig:
    def __init__(self):
        self.batch_size = 2000
        self.gamma = 0.999
        self.actor_lr = 7e-4
        self.critic_lr = 7e-4
        self.gae_lambda = 0.97
        self.policy_clip = 0.1
        self.max_steps = 300 
        self.num_episodes = 9999999999 
        self.train_every = 2000 
        self.n_epochs = 15
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
