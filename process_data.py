import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import median_filter
import numpy as np



def create_multiplot(datasets, title, xlabel, ylabel, legend_labels, output_name, xs=None):
    print(f"Plotting {output_name}...")
    plt.figure(figsize=(6,6))

    if xs is None:
        xs = [np.arange(len(data)) for data in datasets]

    for label, data, x in zip(legend_labels, datasets, xs):
        print(f'\tPlotting {label}')
        plt.plot(x, data, linestyle='-', label=label, fillstyle='none')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    output_path = f"graphics/{output_name}.png"
    plt.savefig(output_path, dpi=500)
    plt.close()
    print(f"Done plotting\n")

def create_plot(data, title, xlabel, ylabel, output_name, x=None, average=0, start_step=0, clip=-100):
    print(f"Plotting {output_name}...")
    plt.figure(figsize=(6,6))
    y = np.array(data)
    np.clip(data, clip, None, out=y) # Clip for better readability
    if x is None:
        x = list(range(start_step, start_step + len(data)))
    plt.plot(x, y)
    # Plot line for averages
    if average > 0 and len(data) > average:
        sums = [np.sum(data[0: average])]
        for i in range(0, len(data) - average - 1):
            sums.append(sums[i] - data[i] + data[i + 1 + average])
        averages = [sums[i] / average for i in range(len(sums))]
        plt.plot(range(average, len(data)), averages, label=f'Mean over Past {average} Steps')
        m, b = np.polyfit(range(len(data)), data, 1) # 1 indicates linear fit
        plt.plot(range(len(data)), m * range(len(data)) + b, label=f'Line of Best Fit')
        plt.legend()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    output_path = f"graphics/{output_name}.png"
    plt.savefig(output_path, dpi=500)
    plt.close()
    print(f"Done plotting\n")

def read_temperature_data(file_path):
        with open(file_path, 'r') as file:
            # Skip the first 4 lines
            data = []
            for line in file:
                try:
                    # Split the line and try to convert the temperature value (9th column, index 8) to float
                    temperature = np.float32(line.strip().split(',')[8])
                    data.append(temperature)
                except (ValueError, IndexError):
                    # If conversion fails or the line doesn't have enough columns, skip this line
                    continue

        return np.array(data)[::30][:200]

def mean_energy_create_plot(data, title, xlabel, ylabel, output_name, x=None, average=0, start_step=0, clip=-100):
    plt.figure(figsize=(6,6))
    if x is None:
        #x = list(range(start_step, start_step + 200))
        l = max([len(ys) for ys in data])
        x = list(range(start_step, start_step + l))

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    plt.plot(x,mean)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.fill_between(x,
                     mean - std,
                     mean + std,alpha=0.2)

    plt.tight_layout()
    output_path = f"graphics/{output_name}.png"
    plt.savefig(output_path, dpi=500)

    plt.title(title)
    plt.close()
    print(f"Done plotting\n")

def plot_active(data, title, xlabel, ylabel, output_name, x=None, average=0, start_step=0, clip=-100):
    plt.figure(figsize=(6,6))
    if x is None:
        x = list(range(start_step, start_step + len(data)))



    mean = np.mean(data, axis=0)[:300]
    std = np.std(data, axis=0)[:300]

    plt.plot(x,mean)
    plt.fill_between(x,
                     mean - std,
                     mean + std,alpha=0.2)

    plt.tight_layout()
    output_path = f"graphics/{output_name}.png"
    plt.savefig(output_path, dpi=500)


    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.close()
    print(f"Done plotting\n")



def plot_active(Repeat):
    label_list = ['No RL', 'RL']

    for status in [0, 1]:
        all_Act = {}
        for re in range(Repeat):
            Act = pickle.load(open('../Data/Log_' + str(re) + '_' + str(status) + '_.p', 'rb'))
            for t in Act.keys():
                if t not in all_Act.keys():
                    all_Act[t] = []
                all_Act[t].append(Act[t])

        mean_Act = {t: np.mean(all_Act[t]) for t in all_Act.keys()}
        strd_Act = {t: np.std(all_Act[t]) for t in all_Act.keys()}

        plt.plot([t for t in sorted(mean_Act.keys())],
                 [mean_Act[t] for t in sorted(mean_Act.keys())],
                 label=label_list[status])

        plt.xlabel('Time', fontsize=16)
        plt.ylabel('Number of active nodes', fontsize=16)
        plt.fill_between([t for t in sorted(mean_Act.keys())],
                         [mean_Act[t] - strd_Act[t] for t in sorted(mean_Act.keys())],
                         [mean_Act[t] + strd_Act[t] for t in sorted(mean_Act.keys())],alpha=0.2)

    plt.legend(prop={'size': 10}, loc='upper right')
    plt.tight_layout()
    plt.savefig('../Plot/Active_med.png', dpi=300)

    plt.show()

datasets = []
labels = []

"""
ids = list(range(17, 35)) 
def remove_spikes(data, threshold=8):
    data = np.array(data)
    #filtered = median_filter(data, size=3)
    filtered = data
    return filtered 

# Load sensor temperature data
for sensor_id in ids:
    try:
        file_path = f'data/towerdataset/tower{sensor_id}Data_processed.csv'
        dataset = read_temperature_data(file_path)
        if(len(dataset) > 0):
            clean = remove_spikes(dataset)
            datasets.append(clean)
            labels.append(f'S{sensor_id}')
        #create_plot(tempdata, f'Sensor {sensor_id} Temperature', 'Step', 'Temperature', f'temp/sensor{sensor_id}temp')
    except:
        print("error")
min_data_len = min((len(data) for data in datasets))
datasets = [data[:min_data_len] for data in datasets] 
minLen = min(len(data) for data in datasets)
xs = np.arange(minLen)
interp_funcs = [scipy.interpolate.interp1d(xs, data, kind='previous', fill_value='extrapolate') for data in datasets]
y = [[interp_func(x/10) for x in range(0, minLen*10)] for interp_func in interp_funcs]
print(len(y[0]))
xs = [np.arange(minLen*10)/200 for _ in y]

# Plot sensor temperature data
create_multiplot(y, 'Sensor Temperature over Simulation Time', 'Time', 'Temperature', labels, f'temp/unfiltered_sensortemps', xs)
exit(0)
"""

#####################################################################
# Load simulation data
rl_agent_log_path = 'rl_agent_figure_data.pkl'
sim_log_path = 'ch0_figure_data.pkl'

with open(rl_agent_log_path, 'rb') as file:
    data = pickle.load(file)
    loss, rewards_log, episode_lengths_log = data

with open(sim_log_path, 'rb') as file:
    data = pickle.load(file)
    messages_log, n_events_detected_log, freq_log, energy_log, throughputs_log = data

s = 10 
# a = 20000
rate_log = freq_log[s:]
reward_log = rewards_log[s:]
throughputs_log = np.stack(throughputs_log)[s:]
energy_log = np.stack(energy_log)[s:]
n_events_detected_log = np.array(n_events_detected_log)[s:]

n_episodes = len(episode_lengths_log)

throughput = [np.sum(t) for t in throughputs_log]
active_nodes = np.array([(t > 0) for t in throughputs_log]).astype(int)
active_nodes = np.sum(active_nodes, axis=1)
energy = [np.sum(e) for e in energy_log]
n_obs = len(energy)

energy_by_episode = []
active_by_episode = []
unique_by_episode = []
start = 0
n_events_detected = n_events_detected_log
print(energy_log)
for i in range(n_obs-1):
    if(max(energy_log[i]) <= 5000 and max(energy_log[i + 1]) >= 9000):
        energy_by_episode.append(energy[start: i+1])
        active_by_episode.append(active_nodes[start: i+1])
        for j in range(start + 1, min(i + 1, len(n_events_detected))):
            n_events_detected[j] += n_events_detected[j-1]
        unique_by_episode.append(n_events_detected[start: i+1])

        start = i + 1

max_episode_length = max([len(e) for e in energy_by_episode])
for i in range(len(energy_by_episode)):
    temp = np.array([0] * max_episode_length)
    l = min(max_episode_length, len(energy_by_episode[i]))
    temp[:l] = energy_by_episode[i][:l]
    temp[l: max_episode_length] = energy_by_episode[i][l-1]
    energy_by_episode[i] = temp
    
    temp = np.array([0] * max_episode_length)
    temp[:l] = active_by_episode[i][:l]
    temp[l:max_episode_length] = active_by_episode[i][l-1]

    active_by_episode[i] = temp[5:]

    temp = np.array([0] * max_episode_length)
    temp[:l] = unique_by_episode[i][:l]
    temp[l: max_episode_length] = unique_by_episode[i][l-1]
    unique_by_episode[i] = temp

energy_by_episode = np.vstack(energy_by_episode)
active_by_episode = np.vstack(active_by_episode)

mean_energy_create_plot(energy_by_episode[-30:], 'Cluster Total Energy Over Episode', 'Step', 'Energy', 'energy_by_e')
mean_energy_create_plot(active_by_episode[-30:], 'Cluster Total Number of Active nodes', 'Step', 'Number of Active Nodes', 'active_by_e')
mean_energy_create_plot(unique_by_episode[-30:], '', 'Step', 'Number of Unique Transmissions', 'unique_by_e')

create_plot(episode_lengths_log, 'Episode Lengths', 'Episode', 'Length (Steps)', 'e_length')
loss = [l.detach().cpu().numpy() for l in loss]
create_plot(loss, 'Loss', 'Training Step', 'Loss', 'loss')
