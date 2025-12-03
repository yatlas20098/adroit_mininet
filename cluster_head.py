import os
import threading
import time
import subprocess
import numpy as np
from mininet.log import setLogLevel, info
from torch_geometric.utils import from_networkx
import pickle
import torch
import math
from scipy.ndimage import median_filter



import networkx as nx
import numpy as np

class Logs():
    def __init__(self, log_every, cluster_idx, sensor_ids):
        self._freq_log = []
        self._energy_log = []
        self._throughputs_log = []
        self._events_detected_log = []
        self._rewards_log = []
        self._step_count = 0
        self._log_every = log_every
        self._cluster_idx = cluster_idx
        self._messages_log = {s_id:[] for s_id in sensor_ids}

    def update_logs(self, messages, events_detected, throughputs, freqs, energy, rewards):
        self._freq_log.append(freqs.detach().cpu().numpy())
        self._energy_log.append(energy.detach().cpu().numpy())
        self._throughputs_log.append(throughputs.detach().cpu().numpy())
        self._rewards_log.append(rewards.detach().cpu().numpy())
        self._events_detected_log.append(events_detected)
        
        for sensor_id, data in messages.items():
            self._messages_log[sensor_id].append(np.mean(data))

        self._step_count += 1

        if self._step_count >= self._log_every:
            self._step_count = 0
            with open(f'ch{self._cluster_idx}_figure_data.pkl', 'wb') as file:
                pickle.dump((self._messages_log, self._events_detected_log, self._freq_log, self._energy_log, self._throughputs_log), file)

class ClusterHead():
    def __init__(self, station, coords, ap, cluster_idx, assigned_sensors, sim_config, sensor_config, event_times, device, log_directory='data/log'):
        self._assigned_sensors = assigned_sensors
        self._assigned_sensor_ids = [s.get_id() for s in assigned_sensors]
        self._n_sensors = len(assigned_sensors)

        self._log_directory = log_directory
        self._cluster_idx = cluster_idx
        self._sim_config = sim_config
        self._sensor_config = sensor_config
        self._device = device
        self._event_times = event_times
        self._prev_energy = torch.tensor([sensor_config.base_energy] * len(self._assigned_sensors), dtype=torch.float32, device=device)
        self._coords = coords
        self._logs = Logs(sim_config.log_every, cluster_idx, self._assigned_sensor_ids)
        self._norm_sensor_dists = self._get_norm_sensor_dists()
        self._prev_event_times = []
        self._events_detected_over_window = {s.get_id():[] for s in assigned_sensors}

        self.__station = station
        self.__ap = ap

        self._pairwise_dists, self._max_dist = self._get_sensor_pairwise_dists()
        
        os.makedirs(log_directory + f'/ch{cluster_idx}_received_data')
        os.makedirs(log_directory + f'/error{cluster_idx}')
        os.makedirs(log_directory + f'/pcaps{cluster_idx}')

    def _get_norm_sensor_dists(self):
        dists = []
        for s in self._assigned_sensors:
            dists.append(self._get_dist(self._coords, s.get_coords()))

        max_dists = max(dists)
        return [d/max_dists for d in dists]

    def _get_sensor_pairwise_dists(self):
        max_dist = 0
        dists = {s.get_id(): {} for s in self._assigned_sensors}
        for i in range(self._n_sensors):
            for j in range(i + 1, self._n_sensors):
                s1 = self._assigned_sensors[i]
                s2 = self._assigned_sensors[j]
                
                dist = self._get_dist(s1.get_coords(), s2.get_coords())
                dists[s1.get_id()][s2.get_id()] = dist
                dists[s2.get_id()][s1.get_id()] = dist

                max_dist = max(max_dist, dist)

        return dists, max_dist
        
    def start(self):
        self._prev_read_time = time.perf_counter()

        thread = threading.Thread(target=self._receive_messages, args=())
        thread.start()

    def recharge(self):
        for s in self._assigned_sensors:
            s.recharge()

    def get_n_sensors(self):
        return len(self._assigned_sensors)

    def get_ap(self):
        return self.__ap

    def get_sensors(self):
        return self._assigned_sensors

    def _get_n_events_detected(self, time_messages_sent):
        event_times_detected = set()
        time_inc = 1 / self._sim_config.data_points_scale
        prev_event_times = set(self._prev_event_times)

        for s, times in time_messages_sent.items():
            rounded_times = np.floor(times / time_inc) * time_inc
            insert_idxs = np.searchsorted(self._event_times[s], rounded_times)
            event_times_idxs = np.abs(self._event_times[s][insert_idxs] - rounded_times) < (time_inc / 2)
            event_times = rounded_times[event_times_idxs] 

            for t in event_times:
                event_times_detected.add(t)

            new_times = event_times_detected - prev_event_times
            self._events_detected_over_window[s].append(len(new_times))
            l = len(self._events_detected_over_window[s])
            if l > self._sim_config.window_size:
                self._events_detected_over_window[s] = self._events_detected_over_window[s][:self._sim_config.window_size]

        new_event_times = event_times_detected - prev_event_times
        print("Event_times_detected: ", new_event_times)
        self._prev_event_times = list(new_event_times) + self._prev_event_times
        if(len(self._prev_event_times) > 10):
            self._prev_event_times = self._prev_event_times[:10]

        return len(new_event_times)
            
    """
    Get an observation of the enviornment.
    
    Args:
        rates (int list): Transmission Frequency index to be used for each sensor

    Returns:
        tuple: (rates_id, similarity, energy, throughputs, reward)
            similarity - num_sensor x num_sensor matrix where entry i,j denotes wheter the data generated by sensor i was similar to the data generated by sensor j
            energy - percent of energy remaining for each sensor
            throughputs - vector of average throughput for each sensor
            reward - reward according to reward function
    """

    def get_observation(self, rates):
        print("Rates: ", rates)
        self._set_rates(rates)
        
        # Wait for observation time
        elapsed_time_since_prev_obs = time.perf_counter() - self._prev_read_time
        sleep_time = self._sim_config.observation_time - elapsed_time_since_prev_obs
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Temperature data is a dict with keys as awake sensor ids and values as the messages received by the cluster head from a sensor
        #temperature_data, throughputs = self._get_temperature_data()
        time_messages_sent, messages, throughputs, awake = self._get_received_messages()

        n_events_detected = self._get_n_events_detected(time_messages_sent)

        # Get list of sensors that had at least one succesfull transmission 
        awake_sensor_ids = [self._assigned_sensors[i].get_id() for i in range(len(self._assigned_sensors)) if awake[i]]
        
        max_freq = max(self._sensor_config.transmission_frequencies)
        max_energy = self._sensor_config.base_energy

        awake = torch.tensor(awake, device=self._device) 
        freqs = torch.tensor([sensor.get_transmit_freq() for sensor in self._assigned_sensors], device=self._device)
        energy = torch.tensor([sensor.get_energy() for sensor in self._assigned_sensors], dtype=torch.float32, device=self._device)
        influence_scores = self._get_influence_scores(awake, energy)
        
        g = nx.Graph()
        g.add_nodes_from([
            (sensor.get_id(), 
            {
                "throughput": (throughputs[i] / max_freq).item(),
                "freq": sensor.get_transmit_freq() / max_freq,
                "energy": sensor.get_energy() / max_energy,
                "dist": self._norm_sensor_dists[i],
                "influence_score": influence_scores[i].item(),
                "events_detected": np.sum(self._events_detected_over_window[sensor.get_id()]) / self._sim_config.window_size
            }) 
            for i, sensor in enumerate(self._assigned_sensors)])

        for node in g.nodes:
            throughput = g.nodes[node]["throughput"]
            freq = g.nodes[node]["freq"]
            e = g.nodes[node]["energy"]
            dist = g.nodes[node]["dist"]
            influence_score = g.nodes[node]["influence_score"]
            events_detected = g.nodes[node]["events_detected"]

            g.nodes[node]["x"] = [throughput, freq, e, dist, influence_score, events_detected]
        
        n_awake = len(awake_sensor_ids)
        for i in range(n_awake - 1):
            s_id1 = awake_sensor_ids[i]
            for j in range(i + 1, n_awake):
                s_id2 = awake_sensor_ids[j]

                dist = self._pairwise_dists[s_id1][s_id2]
                g.add_edge(s_id1,s_id2, weight=dist/self._max_dist)

        #rewards = get_rewards(reward_config, self._config, state)

        total_throughput = throughputs.sum()
        charged_sensor = False
        for s in self._assigned_sensors:
            if s.get_energy() > self._sensor_config.min_energy:
                charged_sensor = True
                break

        terminated = not charged_sensor 
        
        print("Awake: ", awake)
        print(f'Cluster {self._cluster_idx} total throughput over observation: {total_throughput} transmissions/s')
        print("Throughputs: ", throughputs)
        print("Freqs: ", freqs)
        print("Energy: ", energy)

        if total_throughput == 0:
            reward = torch.tensor([-1] * len(self._assigned_sensors), device=self._device)
            state = from_networkx(g).to(self._device)
            state.weight = torch.zeros((0, 1), device=self._device)

            print(f'Reward: {reward}') 
            return (awake, terminated, state, reward)

        # rewards = self._get_reward(state)
        
        rewards = self._get_reward(energy, awake)
        self._logs.update_logs(messages, time_messages_sent, n_events_detected, throughputs, freqs, energy, rewards)

        state = from_networkx(g).to(self._device)
        
        # Handle edge case for graph with 0 or 1 nodes
        if 'weight' not in state:
            state.weight = torch.zeros((0, 1), device=self._device)
        elif state.weight.dim() == 1:
            state.weight = state.weight.unsqueeze(-1)

        return (awake, terminated, state, rewards)
    
    def _percentile_of_value(self, lst, value):
        lst = torch.sort(lst)[0]
        count = lst[lst < value].sum()
        perc = (count / len(lst)) * 100
        return perc
    
    def _get_dist(self, coord1, coord2):
        return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] + coord2[1])**2)

    def _get_influence_scores(self, awake, energy):
        awake_sensor_idxs = torch.argwhere(awake == True)
        delta_energy = (self._prev_energy - energy)
        self._prev_energy = energy
        e_percentiles = {idx.item():self._percentile_of_value(delta_energy[awake_sensor_idxs], delta_energy[idx]) for idx in awake_sensor_idxs}

        influence_scores = torch.zeros_like(energy, device=self._device)
        for i in awake_sensor_idxs:
            s1 = self._assigned_sensors[i]
            for j in awake_sensor_idxs:
                if i == j:
                    continue
                s2 = self._assigned_sensors[j]
                dist = self._get_dist(s1.get_coords(), s2.get_coords())
                
                eps = 1e-6
                influence_scores[i] += e_percentiles[j.item()] / (dist + eps)
        
        rank = torch.unique(influence_scores, sorted=True, return_inverse=True)[1]
        influence_states = torch.zeros_like(influence_scores)
        n = len(awake_sensor_idxs)
        for i in range(n):
            influence_states[i] = rank[i] / n
        print("Influence scores: ", influence_states)
        return influence_states

    def _get_reward(self, energy, awake):
        if(len(awake) == 0):
            return 0

        eps = torch.tensor(1e-8, dtype=torch.float32, device=self._device)
        awake_idxs = awake == True
        awake_energy = energy[awake_idxs]
        reward = torch.tensor([0] * self._n_sensors, dtype=torch.float32, device=self._device)
        
        events_sensed_over_window = torch.tensor([np.sum(n_events_sensed) for n_events_sensed in self._events_detected_over_window.values()], dtype=torch.float32, device=self._device)[awake_idxs]
        events_sensed_norm = (events_sensed_over_window - events_sensed_over_window.mean()) / (events_sensed_over_window.std(unbiased=False) + eps)

        #throughput_norm = (throughput - throughput[awake_idxs].mean()) / (throughput[awake_idxs].std() + eps)
        energy_norm = (awake_energy - awake_energy.mean()) / (awake_energy.std(unbiased=False) + eps)

        #reward[awake_idxs] = torch.exp(events_sensed_norm) * torch.exp(energy_norm)
        reward[awake_idxs] = torch.exp(events_sensed_norm) * torch.exp(energy_norm)

        #reward = torch.tensor([1] * len(self._assigned_sensors), device=self._device)

        return reward

    def _set_rates(self, rates):
        for sensor, rate in zip(self._assigned_sensors, rates):
            sensor.set_rate(rate) 

    """
    Start background netcat listining process for cluster head

    Args:
        sensor_ids (list of sensor objects): A list of sensors the cluster head should receive messages from 
    """
    def _receive_messages(self):
        base_output_file = f'{self._log_directory}/ch{self._cluster_idx}_received_data/sensor'
        ports = []
        
        print("Receiving messages")
        for sensor in self._assigned_sensors:
            # Create a file to store data received from the sensor
            sensor_id = sensor.get_id()
            output_file = f'{base_output_file}_{sensor_id}.txt'
            self.__station.cmd(f'touch {output_file}')

            port = sensor.get_port()
            ports.append(port)

            # Create a listener for sensor i 
            self.__station.cmd(f'stdbuf -o0 nc -n -vv -ul -p {port} -k >> {output_file} 2>> {self._log_directory}/error{self._cluster_idx}/listen_err &')
            info(f"Receiver: Started listening on port {port} for sensor {sensor_id}\n")

        # Capture the network by pcap
        min_port = min(ports)
        max_port = max(ports)

        # pcap_file = f'{self._log_directory}/pcaps/capture.pcap'
        # self.__station.cmd(f'tcpdump -U -i {self.__station.defaultIntf().name} -n udp portrange {min_port}-{max_port} -U -w {pcap_file} &')
        # info(f"Receiver: Started tcpdump capture on ports {min_port}-{max_port}\n")

    def _process_temperature_data_file_line(self, line, data, times):
        # Ignore filler lines
        if line[0] == 'G' or len(line) == 0:
            return
        try:
            # Split the line and try to convert the temperature value (9th column, index 8) to float
            #temperature = np.float32(line.strip().split(',')[8])
            vals = line.strip().split(',')
            time, temperature = np.float32(vals[0]), np.float32(vals[1])
            data.append(temperature)
            times.append(time)
        except (ValueError, IndexError):
            # If conversion fails or the line doesn't have enough columns, skip this line
            pass

    """
    Read data received by cluster head from sensors

    Args:
        file_path (string): Path to file with received data 
        sensor_id (int): ID of the sensor whose data should be read 
(
    Returns:
        String List: List of received packets 
    """
    def _read_messages_from_file(self, file_path):
        bytes_received = 0
        with open(file_path, 'r') as file:
            data = []
            time_sent = []
            for line in file:
                bytes_received += len(line)
                self._process_temperature_data_file_line(line, data, time_sent)
        return np.array(data), np.array(time_sent), bytes_received 

    def _get_received_messages_from_sensor(self, sensor_id):
        file_name = f'sensor_{sensor_id}.txt'

        # Create a copy of the original file
        subprocess.run(["cp", f'{self._log_directory}/ch{self._cluster_idx}_received_data/{file_name}', f'{self._log_directory}/ch{self._cluster_idx}_received_data/.{file_name}'])
        file_path = os.path.join(self._log_directory, f'ch{self._cluster_idx}_received_data/.{file_name}')

        read_time = time.perf_counter()

        # Clear the file for future transmissions
        with open(f'{self._log_directory}/ch{self._cluster_idx}_received_data/{file_name}', 'r+') as file:
            file.truncate(0)
       
        # Read the temperature data for sensor i from the copied file
        data, time_sent, bytes_received = self._read_messages_from_file(file_path)
        observation_time = read_time - self._prev_read_time 
        throughput = bytes_received / (observation_time * self._sensor_config.transmission_size)

        return data, time_sent, throughput, observation_time

    """
    Get the temperature data for all sensors for the previous observation period.  

    Returns:
        temperature_data (dict): a dict with sensors as keys and values as a list of the temprature data transmitted by a sensor 
    """
    def _get_received_messages(self):
        temperature_data = {}
        time_messages_sent = {}
        throughputs = []
        awake = [] 
        
        obs_times = []
        for sensor in self._assigned_sensors:
            s_id = sensor.get_id()
            data, time_sent, throughput, obs_time = self._get_received_messages_from_sensor(s_id)
            obs_times.append(obs_time)
            throughputs.append(throughput)

            if len(data) > 0:
                temperature_data[s_id] = data
                time_messages_sent[s_id] = time_sent
                awake.append(True)
            else:
                awake.append(False)
        
        print("Max True Observation time: ", max(obs_times))
        print("Min True Observation time: ", min(obs_times))

        self._prev_read_time = time.perf_counter() 
        throughputs = torch.tensor(throughputs, device=self._device)
        return time_messages_sent, temperature_data, throughputs, awake

