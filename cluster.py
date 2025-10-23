import sys
import socket
import os
import time
import threading
import random
import multiprocessing
import json
import subprocess
import struct
import numpy as np
import re
import matplotlib.pyplot as plt
import time
import math
import pickle
import os
from torch_geometric.utils import from_networkx

from redundancy_graph import compute_similarity_and_redundancy_graph


from configs import Mininet_Simulation_Config
from reward_config import reward_config
from rewards import get_rewards 
import networkx as nx
from networkx.algorithms import approximation as approx
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
from itertools import combinations
from ortools.linear_solver import pywraplp
import scipy.interpolate 

from tqdm import tqdm
from mininet.node import Controller
from mininet.node import RemoteController
from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.cli import CLI
from mn_wifi.link import wmediumd
from mn_wifi.wmediumdConnector import interference
from datetime import datetime, timedelta
from collections import defaultdict

class Logs():
    def __init__(self, sensor_ids, log_every, cluster_idx):
        self._sensor_ids = sensor_ids
        self._num_sensors = len(sensor_ids) 

        self.rate_log = [[] for _ in range(self._num_sensors)]
        self.throughput_log = [[] for _ in range(self._num_sensors)]
        self.reward_log = {"similarity":[], "throughput":[]}
        self.similarity_reward_log = []
        self.similarity_log = []
        self.throughput_reward_log = []
        self.max_ind_set_log = []
        self.chunks_sent_log = []
        self.transmissions_log = [[] for _ in range(self._num_sensors)]
        self.step_count = 0
        self.chunks_sent = multiprocessing.Array('i', [0] * self._num_sensors)
        self.total_transmissions_received = 0
        self.total_transmissions_received_log = []
        self.log_every = log_every
        self.cluster_idx = cluster_idx


    def update_logs(self, throughputs, similarity, rewards, reward_output, transmission_rates):
        self.similarity_log.append(similarity)
        # Log sensor data and rewards 
        self.chunks_sent_log.append(list(self.chunks_sent))
        for i in range(self._num_sensors):
            self.throughput_log[i].append(throughputs[i])
            # self.energy_log[i].append(self._energy[i])
            self.rate_log[i].append(transmission_rates[i])
            # self._chunks_sent[i] = 0

        if reward_output: 
            max_ind_set = reward_output["throughput"][0]
            print(f"Cluster {self.cluster_idx} Max ind set", [self._sensor_ids[v] for v in max_ind_set])
        else:
            max_ind_set = []

        self.max_ind_set_log.append(max_ind_set)
        self.reward_log["similarity"].append(rewards["similarity"])
        self.reward_log["throughput"].append(rewards["throughput"])
        self.total_transmissions_received_log.append(self.total_transmissions_received)

        # Pickle logs for plotting
        self.step_count += 1
        if self.step_count > self.log_every:
            self.step_count = 0
            with open(f'figure_data{self.cluster_idx}.pkl', 'wb') as file:
                pickle.dump((self._sensor_ids, list(transmission_rates), self.rate_log, self.throughput_log, self.reward_log, self.similarity_reward_log, self.throughput_reward_log, self.max_ind_set_log, self.chunks_sent_log, self.total_transmissions_received_log, self.transmissions_log), file)

class Cluster_Handler():
    def _start_receiver(self):
        info("*** Starting receiver\n")
        print("Starting receiver")
        
        # Activate cluster head listening threads
        receive_threads = []
        receive_thread = threading.Thread(target=self._receive_messages, args=(self._cluster_head,))
        receive_thread.start()
        receive_threads.append(receive_threads)
        return receive_threads

    def _start_senders(self):    
        print("Starting senders")
        info("*** Starting senders\n")
        sender_threads = []
        ch_ip = f'192.168.{self._cluster_idx}.100'
        
        for i, sensor in enumerate(self._sensors):
            # tcpdump_file = f'{self._log_directory}/pcaps/tcpdump_sender_sensor{i}.pcap'
            # sensor.cmd(f'tcpdump -U -i s{i}-wlan0 -w {tcpdump_file} &')
            
            thread = threading.Thread(target=self._send_messages_to_cluster_head, args=(sensor, ch_ip, i))
            thread.start()
            sender_threads.append(thread)

    """
    Set sensor transmission rates 

    Args:
        new_rates (integer list): List of indicies for new sensor transmission rates 
    """
    def set_rates(new_rates):
        for i in range(len(new_rates)):
            self._transmission_rates[i] = new_rates[i]
    
    """
    Interpolate dataset using linear splines.
    Note: A second in the spline corresponds to 250 data points in the dataset file.

    Args:
        dataset_dir (string): Path to dataset file
        sensor_id (int): ID of the sensor associcated to the dataset file
        file_lines_per_transmission (int): How many file lines to include per transmission 

    Returns:
        scipy function: function for interpolated data 
    """

    def _interpolate_dataset(self, dataset_dir, maxlen=100000000):
        # Cache the dataset into memory 
        with open(dataset_dir, 'r') as file:
            lines = file.readlines()

            # Skip the first 4 lines
            data = []
            for line in lines:
                if len(data) > maxlen:
                    break
                try:
                    # Split the line and try to convert the temperature value (9th column, index 8) to float
                    temperature = np.float32(line.strip().split(',')[8])
                    data.append(temperature)
                except (ValueError, IndexError):
                    # If conversion fails or the line doesn't have enough columns, skip this line
                    continue

            xs = np.arange(len(data))/(240)
            interp_func = scipy.interpolate.interp1d(xs, data)
            self._min_dataset_t = min(len(data)/(240), self._min_dataset_t)
            return interp_func

    def _get_sensor_temperature_data(self, sensor_idx):
        file_name = f'sensor_{self._config.sensor_ids[sensor_idx]}.txt'

        # Create a copy of the original file
        subprocess.run(["cp", f'{self._log_directory}/ch{self._cluster_idx}_received_data/{file_name}', f'{self._log_directory}/ch{self._cluster_idx}_received_data/.{file_name}'])
        file_path = os.path.join(self._log_directory, f'ch{self._cluster_idx}_received_data/.{file_name}')
        curr_time = time.perf_counter()
        true_observation_period = (curr_time - self._prev_obs_end_time[sensor_idx])
        self._prev_obs_end_time[sensor_idx] = curr_time

        # Clear the file for future transmissions
        with open(f'{self._log_directory}/ch{self._cluster_idx}_received_data/{file_name}', 'r+') as file:
            file.truncate(0)
       
        # Read the temperature data for sensor i from the copied file
        data, throughput = self._read_temperature_data_from_file(file_path, sensor_idx)
        
        throughput = throughput / true_observation_period
        return data, throughput

    """
    Get the temperature data for all sensors for the previous observation period.  

    Returns:
        temperature_data (dict): a dict with sensors as keys and values as a list of the temprature data transmitted by a sensor 
    """
    def _get_temperature_data(self):
        temperature_data = {}
        throughputs = []

        for i in range(self._num_sensors):
            data, throughput = self._get_sensor_temperature_data(i)
            throughputs.append(throughput)
            if len(data) > 0:
                temperature_data[i] = data
        
        return temperature_data, throughputs

    def _update_rates(self, rates):
        for i in range(self._num_sensors):
            self._transmission_rates[i] = int(rates[i])

        
    def _similarity_changed(self, new_similarity):
        if self._prev_similarity is None or self._t_since_last_sim_change == 0:
            self._prev_similarity = new_similarity
            self._t_since_last_sim_change += 1
            return False 

        for i in range(self._num_sensors - 1):
            for j in range(i, self._num_sensors):
                if(new_similarity[i][j] != self._prev_similarity[i][j]):
                    self._prev_similarity = new_similarity
                    self._t_since_last_sim_change = 0 

                    return True
        
        self._t_since_last_sim_change += 1
        return False 


    """
    Get an observation of the enviornment.
    
    Args:
        rates (int list): Transmission Frequency index to be used for each sensor
        heuristic (bool): If heursitic is True, rates are discared and a heursitic policy is used to assign transmission rates
        replay (int): If replay is greater than -1, the similarity matrix is retrived from step replay to calculate rewards  

    Returns:
        tuple: (rates_id, similarity, energy, throughputs, reward)
            similarity - num_sensor x num_sensor matrix where entry i,j denotes wheter the data generated by sensor i was similar to the data generated by sensor j
            energy - percent of energy remaining for each sensor
            throughputs - vector of average throughput for each sensor
            reward - reward according to reward function
    """

    def get_observation(self, rates):
        self._update_rates(rates)

        # Wait for observation time
        time.sleep(self._config.observation_time)
        
        # Temperature data is a dict with keys as awake sensor ids and values as the data received by the cluster head from a sensor
        #temperature_data, throughputs = self._get_temperature_data()
        temperature_data, throughputs = self._cluster_head.get_temperature_data()
                
        # Get similarity matrix and redudancy graph
        transmit_freqs = [self._config.transmission_frequencies[int(r)] for r in rates]

        similarity, connectivity_graph, redundancy_graph = compute_similarity_and_redundancy_graph(self._config, transmit_freqs, temperature_data, throughputs, replay)
        total_throughput = np.sum(throughputs)

        terminated = self._similarity_changed(similarity)

        print(f'Cluster {self._cluster_idx} total throughput over observation: {total_throughput} transmissions/s')
        
        if total_throughput == 0:
            rewards = {
                    "throughput": -10,
                    "similarity": [-5]*self._num_sensors
                    }

            self._logs.update_logs(throughputs, similarity, rewards, None, self._transmission_rates)
            redundancy_graph = from_networkx(redundancy_graph)
            connectivity_graph = from_networkx(connectivity_graph)


            return (terminated, connectivity_graph, redundancy_graph, rewards)

        rewards, reward_output = get_rewards(reward_config, self._config, redundancy_graph)
        self._logs.update_logs(throughputs, similarity, rewards, reward_output, self._transmission_rates)

        redundancy_graph = from_networkx(redundancy_graph)
        connectivity_graph = from_networkx(connectivity_graph)

        return (terminated, connectivity_graph, redundancy_graph, rewards)

    def _save_sensor_temps(self):
        while True:
            with open(f'sensortemps{self._cluster_idx}.pkl', 'wb') as file:
                pickle.dump((self._transmissions_log), file)
            time.sleep(20)

    def __init__(self, cluster_head, sensors, cluster_idx, simulation_config, log_directory='data/log', dataset_directory='data/towerdataset'):
        self._config = simulation_config
        self._logs = Logs(self._config.sensor_ids, self._config.log_every, cluster_idx)
        self._num_sensors = len(self._config.sensor_ids)
        self._cluster_idx = cluster_idx
        self._throughputs = [0 for i in range(self._num_sensors)]
        self._total_throughput = 0
        self._max_total_throughput = 0
        self._prev_obs_end_time = multiprocessing.Array('d', [time.perf_counter()] * self._num_sensors)
        self._offset = random.randint(50000, 100000)
        self._n_train_steps = 0
        self._min_dataset_t = 99999999
        self._prev_similarity = None
        self._t_since_last_sim_change = 999999999
        
        self._cluster_head = cluster_head
        self._sensors = sensors

        # Directories
        self._log_directory = log_directory
        self._dataset_directory = dataset_directory
       
        # Rate configuration
        self._transmission_rates = multiprocessing.Array('i', [0] * self._num_sensors)
        #self._transmission_frequencies = np.array([20, 40, 60, 80]) # Possible number of times a sensor can transmit per frame

        # Energy configuration
        self._full_energy = 100
        #self._recharge_time = 10 * self._config.transmission_frame_duration
        self._recharge_time = 10
        self._recharge_threshold = 20
        self._energy = [self._full_energy for _ in range(self._num_sensors)]

        # Data logs 
                #self.chunks_sent = [0 for _ in range(self._num_sensors)] # List of the number of transmissions sent by each sensor

        # Delete the log folder if already it exists
        # subprocess.run(["rm", "-rf", log_directory])

        # Create log directories 
        # os.makedirs(log_directory)
        os.makedirs(log_directory + f'/ch{self._cluster_idx}_received_data')
        os.makedirs(log_directory + f'/error{self._cluster_idx}')
        os.makedirs(log_directory + f'/pcaps{self._cluster_idx}')

        if not os.path.exists(dataset_directory):
            print("Dataset directory does not exist. Exiting now")
            exit()

        # Initalize the dataset for each sensor 
        self._datasets = {}
        for i in range(self._num_sensors):
            tower_number = self._config.sensor_ids[i]
            file_path = f'{dataset_directory}/tower{tower_number}Data_processed.csv'
            if os.path.exists(file_path):
                #self._datasets[i] = self._preprocess_dataset_into_chunks(file_path, i, file_lines_per_chunk)
                self._datasets[i] = self._interpolate_dataset(file_path)
            else:
                print(f"Warning: Dataset file not found for sensor {i}: {file_path}")
                self._datasets[i] = []

        self._simulation_start_time = time.perf_counter()
        # self._create_topology()

    
    """
    Kill netcat listining process for cluster head

    Args:
        node (station): cluster head 
    """
    def _stop_receivers(self, node):
        # Kill the receiver
        node.cmd('pkill -f "nc -ul"')
        # Kill the tcpdump capture
        node.cmd('pkill tcpdump')
        info("Stopped all nc receivers and tcpdump\n")

    """
    Begin message transmission from sensors to cluster head and reap all threads after transmission concludes.   
    """
    def start(self):
        print("STARTING")
        
        info("*** Setting up communication flow\n")
        try:
            receive_thread = self._start_receiver()

            # Give listening thread time to start
            time.sleep(10) 
            
            # Start senders
            self._start_time = time.perf_counter()
            self._start_senders()

            #plot_thread = threading.Thread(target=self._save_sensor_temps, args=())
            #plot_thread.start()
            #plot_thread.join()

            print("Waiting for senders to finish")
            for thread in sender_threads:
                thread.join()

            self._stop_receivers(cluster_head)
            receive_thread.join()

            for sensor in sensors:
                sensor.cmd('pkill tcpdump')
            cluster_head.cmd('pkill nc')

        except Exception as e:
            info(f"*** Error occurred during communication: {str(e)}\n")
            
        #info("*** Running CLI\n")
        #CLI(self._net)

        #info("*** Stopping network\n")
        #self._net.stop()

if __name__== '__main__':
    sensor_ids = range(5,15)
    observation_time = 1
    transmission_size = 2*1024
    transmission_frame_duration = 1
    file_lines_per_chunk = 1
    num_transmission_frames = 3000
    sim_config = Mininet_Simulation_Config(sensor_ids=sensor_ids, observation_time=observation_time, transmission_size=transmission_size, transmission_frame_duration=transmission_frame_duration, file_lines_per_chunk=file_lines_per_chunk, num_transmission_frames=num_transmission_frames)
    

    cluster = Cluster(0, sim_config, log_directory=f'data/log')
    # cluster.establish_connection_with_rl_agent()
    print("Starting cluster")
    cluster_thread = threading.Thread(target=cluster.start, args=())
    cluster_thread.start()
    

    # receive_rates_thread = threading.Thread(target=cluster.receive_rates_from_rl_agent, args=())
    # receive_rates_thread.start()

    cluster_thread.join()
    #receive_rates_thread.join()

