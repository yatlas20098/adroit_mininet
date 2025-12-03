from mininet.node import Controller
from mininet.node import RemoteController
from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.cli import CLI
from mn_wifi.link import wmediumd
from mn_wifi.wmediumdConnector import interference

import os
import numpy as np
import scipy
import time

from sensor import Sensor
from cluster_head import ClusterHead 

def _get_tower_coords(center_tower_id, dataset_directory='data/towerdataset'):
    import csv

    tower_coords_long_lat = {}
    with open(f'{dataset_directory}/tower_coords.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader) # Skip header row
        for line in reader:
            lat = float(line[0])
            long = float(line[1])
            tower_id = int(line[2][4:])

            tower_coords_long_lat[tower_id] = (long, lat)

    tower_coords_cartesian = _long_lat_to_cartesian(center_tower_id, tower_coords_long_lat)

    return tower_coords_cartesian

def _long_lat_to_cartesian(center_tower_id, tower_coords):
    from pyproj import Transformer

    first_lon, first_lat = tower_coords[center_tower_id]
    
    zone_number = int((first_lon + 180) / 6) + 1
    hemisphere = 'north' if first_lat >= 0 else 'south'

    if hemisphere == 'north':
        epsg_code = 32600 + zone_number
    else:
        epsg_code = 32700 + zone_number

    transformer = Transformer.from_crs("epsg:4326", f"epsg:{epsg_code}", always_xy=True)
    

    origin_x, origin_y = transformer.transform(first_lat, first_lon)

    tower_coords_cartesian = {tower_id: transformer.transform(lat, lon) for tower_id, (lon, lat) in tower_coords.items()}
    tower_coords_cartesian_normalized = {tower_id: ((x - origin_x)/120, (y - origin_y)/120) for tower_id, (x,y) in tower_coords_cartesian.items()}

    return tower_coords_cartesian_normalized

def _get_datasets(sensor_ids, data_points_scale, dataset_directory='data/towerdataset'):
    datasets = {}
    event_times = {}
    min_dataset_len = float('inf')
    for sensor_id in sensor_ids:
        file_path = f'{dataset_directory}/tower{sensor_id}Data_processed.csv'
        if os.path.exists(file_path):
            datasets[sensor_id], dataset_length_s, event_times[sensor_id] = _interpolate_dataset(file_path, data_points_scale)
            min_dataset_len = min(min_dataset_len, dataset_length_s)
        else:
            print(f"Warning: Dataset file not found for sensor {sensor_id}: {file_path}")
            datasets[sensor_id] = []

    return datasets, min_dataset_len, event_times

def _get_dist(coord1, coord2):
        return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] + coord2[1])**2)

def create_mininet_network(sim_config, sensor_config, device):
    sim_start_time = time.perf_counter()
    cluster_heads = []
    
    tower_coords = _get_tower_coords(1)

    net = Mininet_wifi(controller=Controller, link=wmediumd, wmediumd_mode=interference)
    for cluster_idx, (cluster_head_id, sensor_ids) in enumerate(sim_config.clusters):
        cluster_head_ip, cluster_head_station, sensor_stations, ap = _create_mininet_cluster(net, tower_coords, cluster_idx, cluster_head_id, sensor_ids)
        datasets, min_dataset_len, event_times = _get_datasets(sensor_ids, sim_config.data_points_scale)

        cluster_head_coords = tower_coords[cluster_head_id] 
        sensors = []
        for i, sensor_id in enumerate(sensor_ids):
            sensor_coords = tower_coords[sensor_id]
            dist_from_ch = _get_dist(cluster_head_coords, sensor_coords)
            s = Sensor(sensor_id, sensor_coords, dist_from_ch, sensor_stations[i], cluster_head_ip, cluster_idx, sensor_config, datasets[sensor_id], min_dataset_len, sim_start_time)
            sensors.append(s)
        
        ch = ClusterHead(cluster_head_station, cluster_head_coords, ap, cluster_idx, sensors, sim_config, sensor_config, event_times, device)
        cluster_heads.append(ch)

    info("*** Adding Controller\n")
    c0 = net.addController(f'c0', controller=Controller, link=wmediumd, wmedium_mode=interference)

    info("*** Configuring wifi nodes\n")
    net.configureWifiNodes()

    net.build()
    net.start()

    for cluster_head in cluster_heads:
        ap = cluster_head.get_ap()
        ap.start([c0])

        for sensor in cluster_head.get_sensors():
            station = sensor.get_station()
            station.setAssociation(ap)
            iface = station.params['wlan'][0]
            station.cmd(f'iw dev {iface} set retry short 0')

    # Start cluster heads
    for cluster_head in cluster_heads:
        print("Starting cluster head")
        cluster_head.start() 
    
    # Give cluster heads time to listen for transmissions
    time.sleep(10)

    # Start sensors
    for cluster_head in cluster_heads:
        sensors = cluster_head.get_sensors()
        for sensor in sensors:
            sensor.start()

    return cluster_heads

"""
Interpolate dataset using linear splines.
Note: A second in the spline corresponds to 250 data points in the dataset file.

Args:
    dataset_dir (string): Path to dataset file
    datapoints_per_s (int): How many datapoints should be included in a second

Returns:
    scipy function: function for interpolated data 
"""
def _interpolate_dataset(dataset_dir, data_points_scale, maxlen=100000000):
    # from scipy.ndimage import median_filter

    # Cache the dataset into memory 
    with open(dataset_dir, 'r') as file:
        lines = file.readlines()

        # Skip the first 4 lines
        data = []
        xs = []
        prev_t = 0
        n_roll_over = 0

        for line in lines:
            if len(data) > maxlen:
                break
            try:
                # Split the line and try to convert the temperature value (9th column, index 8) to float
                row = line.strip().split(',')
                t = np.float32(row[0])
                if t < prev_t:
                    n_roll_over += 1

                temperature = np.float32(row[8])
                data.append(temperature)
                sec_in_hr = 3600
                xs.append(t + (n_roll_over*sec_in_hr))

                prev_t = t

            except (ValueError, IndexError):
                # If conversion fails or the line doesn't have enough columns, skip this line
                continue
        
        # Remove sharp jumps
        # data = median_filter(data, size=3)

        # xs = np.arange(len(data)) / (data_points_per_s)
        xs = np.array(xs) / data_points_scale
        interp_func = scipy.interpolate.interp1d(xs, data, kind='previous', fill_value='extrapolate')
        dataset_length_s = max(xs)
        event_times = _get_event_times(data, xs)

        return interp_func, dataset_length_s, event_times

def _get_event_times(data, xs):
    data = np.array(data)
    event_idx = []

    means = [np.sum(data[:10])]
    for i in range(10, len(data)):
        means.append(means[-1] + data[i] - data[i - 10])
    means = np.array(means) / 10

    for i in range(10, len(data) - 10):
        l_mean = means[i - 10]
        r_mean = means[i + 1]


        diff_lr = abs(l_mean - r_mean)
        diff_center = abs(2*data[i] - l_mean - r_mean)
        if diff_lr < 0.05 and diff_center > 0.2:
            event_idx.append(i)
    event_idx = np.array(event_idx)
    if len(event_idx) == 0:
        return []

    event_times = xs[event_idx]
    return event_times 

def _create_mininet_cluster(net, tower_coords, cluster_idx, cluster_head_id, sensor_ids):
        channel = 6

        # Convert cluster idx to 6 digit hex
        cluster_idx_hex = '{:06x}'.format(cluster_idx)
        
        # Devote first half of mac address to cluster idx
        mac_prefix = ':'.join([cluster_idx_hex[i:i+2] for i in range(0,6,2)])
        
        # create accesspoint
        cluster_head_cords = tower_coords[cluster_head_id]
        ap = net.addAccessPoint(f'ap{cluster_idx}', 
                                 ssid=f'ssid-ap{cluster_idx}', 
                                 mac=f'{mac_prefix}:FF:FF:FF',
                                 mode='g', 
                                 channel=f'{channel}', 
                                 position=f'{cluster_head_cords[0] + 50},{cluster_head_cords[1]}, 0')
       
        # create cluster head station 
        cluster_head_ip = f'192.168.{cluster_idx}.100'
        cluster_head_station = net.addStation(f'ch{cluster_idx}', 
                                            ip=cluster_head_ip + "/24",
                                            mac=f'{mac_prefix}:EF:FF:FF',
                                            range='150', 
                                            position=f'{cluster_head_cords[0]},{cluster_head_cords[1]},0')

        # create sensor stations 
        sensor_stations = []
        for i in sensor_ids:
            cord = tower_coords[i]

            idx_hex = '{:06x}'.format(i + 1)
            idx_mac = ':'.join([idx_hex[i:i+2] for i in range(0,6,2)])

            sensor_mac = f'{mac_prefix}:{idx_mac}'
            ip_address = f'192.168.{cluster_idx}.{i + 2}/24'
            sensor_station = net.addStation(f's-{cluster_idx}-{i}', 
                                        ip=ip_address,
                                        mac=sensor_mac,
                                        range='116', 
                                        position=f'{cord[0]},{cord[1]},0')
            sensor_stations.append(sensor_station)
        return cluster_head_ip, cluster_head_station, sensor_stations, ap
