import numpy as np
import time
from configs import SensorConfig 
import threading
from mininet.log import setLogLevel, info

class Sensor():
    def __init__(self, sensor_id, cords, dist_from_cluster_head, station, cluster_head_ip, cluster_idx, config, dataset, dataset_max_time, sim_start_time, log_directory='data/log'):
        self._config = config 
        self._transmit_freq = self._config.transmission_frequencies[0]
        self._energy = config.base_energy
        self._cluster_head_ip = cluster_head_ip
        self._sim_start_time = sim_start_time
        self._chunks_sent = 0
        self._coords = cords
        self._dist_from_cluster_head = dist_from_cluster_head
        
        self.__station = station
        self.__dataset = dataset
        self.__log_directory = log_directory
        self.__id = sensor_id
        self.__cluster_idx = cluster_idx
        self.__dataset_max_time = dataset_max_time

        self.__port = sensor_id + cluster_idx*100 

        # Create filler used to pad transmissions to full size
        self.__filler = 'G' * (self._config.transmission_size)
        
    def start(self):
        thread = threading.Thread(target=self._send_messages_to_cluster_head, args=())
        thread.start()

    def recharge(self):
        self._energy = self._config.base_energy

    def set_rate(self, rate):
        self._transmit_freq = self._config.transmission_frequencies[rate]

    def get_id(self):
        return self.__id

    def get_port(self):
        return self.__port

    def get_energy(self):
        return self._energy

    def get_station(self):
        return self.__station

    def get_transmit_freq(self):
        return self._transmit_freq

    def get_coords(self):
        return self._coords 

    def _compute_energy_consumption(self, transmission_dist):
        aE, kappaE, baseline_energy_dist, baseline_energy_consumption = self._config.aE, self._config.kappaE, self._config.baseline_energy_dist, self._config.baseline_energy_consumption

        dist_diff = (transmission_dist - baseline_energy_dist)
        a = 1 + np.exp(-aE * dist_diff)
        energy_consumption = baseline_energy_consumption + (kappaE / a)

        return energy_consumption

    def _update_energy(self, transmission_dist):
        energy_consumption = self._compute_energy_consumption(transmission_dist)
        self._energy -= energy_consumption
    
    def _send_message(self, ip, dist):
        transmission_start_time = time.perf_counter()
        time_since_sim_start = transmission_start_time - self._sim_start_time
        rolled_over_time = time_since_sim_start % (self.__dataset_max_time - 1) 
        datapoint = f'{self.__dataset(rolled_over_time):.4f}' 
        str_time = f'{rolled_over_time:.5f}'

        nc_error_log_file = f'{self.__log_directory}/error{self.__cluster_idx}/nc{self.__id}'
        padded_message = f"\n{str_time},{datapoint}\n{self.__filler[len(datapoint) + len(str_time) + 8:]}\n"
        cmd = f'echo "{padded_message}" | nc -v -w0 -u {ip} {self.__port} >> {nc_error_log_file} 2>&1 &'
        self.__station.cmd(cmd)
        # TODO: Compute dist
        self._update_energy(dist)

        self._chunks_sent += 1

        transmission_time = time.perf_counter() - transmission_start_time
        return transmission_time 
        
    def _recharge(self):
        charge_count += 1

        recharge_time = time.time()
        rechar_time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(recharge_time))
        #info(f'Sensor {sensor_id}: Energy {self._energy[sensor_id]} below threshold ({self._recharge_threshold}). Recharging...')
        time.sleep(self._recharge_time)

        # Update the sensors energy and log the new energy level
        self._energy[sensor_idx] = self._full_energy
        sensor.energy = energy

        #info(f'Sensor {sensor_idx}: Energy Recharged to full energy ({self._full_energy}), Current time: {recharge_time}, Charge Count: {charge_count}. Resuming operations.')
        
        #if next_chunk_idx >= len(chunks):
            #self._transmit_data_status[sensor_idx].set()
        #    break


    """
    Send message from a sensor to the cluster head

    Args:
        sensor (station): sensor to send message from
        ch_ip (string): cluster head ip
        sensor_idx (int): index of sensor
    """
    def _send_messages_to_cluster_head(self):
        # Return if there is not data to form messages with
        if not self.__dataset:
            info(f"Sensor {self.__id}: No data available. Skipping send_messages.\n")
            return
        
        while True:
            # TODO: How does a sensor recharge?
            # Recharge sensor if energy is below the recharge threshold?
            if self._energy <= self._config.min_energy:
                sleep_time = 1 / min(self._config.transmission_frequencies)
                time.sleep(sleep_time)
                continue

            #if self._energy[sensor_idx] < self._recharge_threshold:
                
            # Sensor should skip tranmission during the current frame
            if self._transmit_freq == 0:
                sleep_time = 1 / min(self._config.transmission_frequencies)
                time.sleep(sleep_time)
            else:
                transmission_time = self._send_message(self._cluster_head_ip, self._dist_from_cluster_head)
                sleep_time = (1 / self._transmit_freq) - transmission_time
                if(sleep_time > 0):
                    time.sleep(sleep_time)
        info(f"Sensor {self.__id}: Finished sending messages\n")
