import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.stats._qmc import LatinHypercube


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sample_X0(Ns, num_states):
    '''Uniform sampling from the initial condition domain.'''
    N = Ns
    # bounds = np.hstack((self.X0_lb, self.X0_ub))
    # D = bounds.shape[0]
    sampler = LatinHypercube(d=num_states)
    x0_1 = sampler.random(n=N)
    x0_1 = x0_1/np.sum(x0_1, axis=1).reshape(-1, 1)
    x0_2 = sampler.random(n=N)
    x0_2 = x0_2/np.sum(x0_2, axis=1).reshape(-1, 1)
    # x0_1 = 0.25 * np.ones((N, self.X0_lb.shape[0]))
    # X0 = np.concatenate((x0_1, x0_1), axis=1) # concatenate into N x 8
    X0_1 = np.concatenate((x0_1, x0_2), axis=1) # concatenate into N x 8

    X0_2 = np.concatenate((x0_2, x0_1), axis=1)

    X0 = np.concatenate((X0_1, X0_2), axis=0)

    return X0.astype(np.float32)

def sample_X0_new(Ns, num_states):
    '''Uniform sampling from the initial condition domain.'''
    N = Ns
    # bounds = np.hstack((self.X0_lb, self.X0_ub))
    # D = bounds.shape[0]
    sampler = LatinHypercube(d=num_states)
    x0_1 = sampler.random(n=N)
    x0_1 = x0_1/np.sum(x0_1, axis=1).reshape(-1, 1)
    x0_2 = sampler.random(n=N)
    x0_2 = x0_2/np.sum(x0_2, axis=1).reshape(-1, 1)
    # x0_1 = 0.25 * np.ones((N, self.X0_lb.shape[0]))
    # X0 = np.concatenate((x0_1, x0_1), axis=1) # concatenate into N x 8
    X0 = np.concatenate((x0_1, x0_2), axis=1) # concatenate into N x 8

    return X0.astype(np.float32)

def sample_X0_uniform(Ns, num_states):
    '''Uniform sampling from the initial condition domain.'''
    N = Ns
    # bounds = np.hstack((self.X0_lb, self.X0_ub))
    # D = bounds.shape[0]
    x0_1 = np.random.uniform(0, 1, (N, num_states))
    x0_1 = x0_1/np.sum(x0_1, axis=1).reshape(-1, 1)
    x0_2 = np.random.uniform(0, 1, (N, num_states))
    x0_2 = x0_2/np.sum(x0_2, axis=1).reshape(-1, 1)
    # x0_1 = 0.25 * np.ones((N, self.X0_lb.shape[0]))
    # X0 = np.concatenate((x0_1, x0_1), axis=1) # concatenate into N x 8
    X0 = np.concatenate((x0_1, x0_2), axis=1) # concatenate into N x 8

    return X0.astype(np.float32)




def softmax(x, alpha):
    return torch.exp(alpha * x)/torch.sum(torch.exp(alpha * x), dim=1).reshape(-1, 1)


def boltzmann_operator(x, alpha):
    return sum(x * np.exp(alpha * x))/sum(np.exp(alpha * x))

def boltzmann_vec(x, alpha):
    return torch.sum(x * torch.exp(alpha * x), dim=1) / torch.sum(torch.exp(alpha * x), dim=1)

def plot_2d(x1_values, x2_values, time_steps):
    # Define the bounding box sizes for each region
    box_sizes = [1, 1]

    # Multiply proportions by the total number of swarms
    total_swarms = 1000
    x1_counts = [[int(prop * total_swarms) for prop in timestep] for timestep in x1_values]
    x2_counts = [[int(prop * total_swarms) for prop in timestep] for timestep in x2_values]

    # Create subplots for each time step
    fig, axs = plt.subplots(len(time_steps), 2, figsize=(12, 4 * len(time_steps)))

    # Generate plots for each time step
    for t in range(len(time_steps)):
        # axs[t].set_xlim(-2, 2)
        # axs[t].set_ylim(-2, 2)
        # axs[t, 0].set_xlabel('X-axis')
        # axs[t, 0].set_ylabel('Y-axis')
        axs[t, 0].set_title(f'Time Step {t + 1}, t = {time_steps[t]}')
        axs[t, 0].xaxis.set_tick_params(labelbottom=False)
        axs[t, 0].yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        axs[t, 0].set_xticks([])
        axs[t, 0].set_yticks([])

        # Plot bounding boxes for each region
        corners = [(0, 0), (0, 2 * max(box_sizes))]
        color1 = 'red'
        color2 = 'blue'

        for i, corner in enumerate(corners):
            rectangle = Rectangle(corner, box_sizes[i], box_sizes[i], edgecolor='black', facecolor='none')
            axs[t, 0].add_patch(rectangle)

            # Plot dots representing swarms in each region
            if x1_counts[t][i] > x2_counts[t][i]:
                alpha1 = 1
                alpha2 = 0.3
            else:
                alpha2 = 1
                alpha1 = 0.3
            x1_swarm_dots = np.random.uniform(corner[0], corner[0] + box_sizes[i], x1_counts[t][i])
            y1_swarm_dots = np.random.uniform(corner[1], corner[1] + box_sizes[i], x1_counts[t][i])
            axs[t, 0].scatter(x1_swarm_dots, y1_swarm_dots, color=color1, alpha=alpha1)

            x2_swarm_dots = np.random.uniform(corner[0], corner[0] + box_sizes[i], x2_counts[t][i])
            y2_swarm_dots = np.random.uniform(corner[1], corner[1] + box_sizes[i], x2_counts[t][i])
            axs[t, 0].scatter(x2_swarm_dots, y2_swarm_dots, color=color2, alpha=alpha2)

            # Add region labels
            if i == 0:  # region 1
                axs[t, 0].annotate("Region 1", (corner[0] + box_sizes[i] / 2, corner[1] + 1.1 * box_sizes[i]),
                                   ha='center', va='center')
            else:  # region 2
                axs[t, 0].annotate("Region 2", (corner[0] + box_sizes[i] / 2, corner[1] - box_sizes[i] / 4),
                                   ha='center', va='center')
        # Plot bar chart comparing the number of swarms in each region
        regions = ['Region 1', 'Region 2']
        counts_x1 = x1_counts[t]
        counts_x2 = x2_counts[t]
        bar_width = 0.35

        x = np.arange(len(regions))
        axs[t, 1].bar(x, counts_x1, width=bar_width, label='Attackers', color='red', alpha=0.5)
        axs[t, 1].bar(x + bar_width, counts_x2, width=bar_width, label='Defenders', color='blue', alpha=0.5)

        axs[t, 1].set_xticks(x + bar_width / 2)
        axs[t, 1].set_xticklabels(regions)
        axs[t, 1].set_ylabel('Number of Swarms')
        axs[t, 1].set_title('Number of Swarms in Each Region')
        axs[t, 1].legend()
    plt.tight_layout()
    plt.show()


def check_target_status_2d(X1, X2, beta):
    """

    :param state: combined state of the system
    :return: True if target reached
    """
    diff = X1 - X2
    return (np.max(X1 - X2, axis=1) - beta) >= 0


def plot_4d(x1_values, x2_values, time_steps):
    # Define the bounding box sizes for each region
    box_sizes = [1, 1, 1, 1]

    # Multiply proportions by the total number of swarms
    total_swarms = 1000
    x1_counts = [[int(prop * total_swarms) for prop in timestep] for timestep in x1_values]
    x2_counts = [[int(prop * total_swarms) for prop in timestep] for timestep in x2_values]

    # Create subplots for each time step
    fig, axs = plt.subplots(len(time_steps), 2, figsize=(12, 4 * len(time_steps)))

    # Generate plots for each time step
    for t in range(len(time_steps)):
        # axs[t].set_xlim(-2, 2)
        # axs[t].set_ylim(-2, 2)
        # axs[t, 0].set_xlabel('X-axis')
        # axs[t, 0].set_ylabel('Y-axis')
        axs[t, 0].set_title(f'Time Step {t + 1}, t = {time_steps[t]}')
        axs[t, 0].xaxis.set_tick_params(labelbottom=False)
        axs[t, 0].yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        axs[t, 0].set_xticks([])
        axs[t, 0].set_yticks([])

        # Plot bounding boxes for each region
        corners = [(0, 0), (0, 2 * max(box_sizes)), (2 * max(box_sizes), 2 * max(box_sizes)), (2 * max(box_sizes), 0)]
        color1 = 'red'
        color2 = 'blue'

        for i, corner in enumerate(corners):
            rectangle = Rectangle(corner, box_sizes[i], box_sizes[i], edgecolor='black', facecolor='none')
            axs[t, 0].add_patch(rectangle)

            # Plot dots representing swarms in each region

            # alpha1 = x1_counts[t][i]/1000
            # alpha2 = x2_counts[t][i]/1000
            if x1_counts[t][i] > x2_counts[t][i]:
                alpha1 = 1
                alpha2 = 0.3
            else:
                alpha2 = 1
                alpha1 = 0.3
            x1_swarm_dots = np.random.uniform(corner[0], corner[0] + box_sizes[i], x1_counts[t][i])
            y1_swarm_dots = np.random.uniform(corner[1], corner[1] + box_sizes[i], x1_counts[t][i])
            axs[t, 0].scatter(x1_swarm_dots, y1_swarm_dots, color=color1, alpha=alpha1)

            x2_swarm_dots = np.random.uniform(corner[0], corner[0] + box_sizes[i], x2_counts[t][i])
            y2_swarm_dots = np.random.uniform(corner[1], corner[1] + box_sizes[i], x2_counts[t][i])
            axs[t, 0].scatter(x2_swarm_dots, y2_swarm_dots, color=color2, alpha=alpha2)

            # Add region labels
            if i == 0 or i == 3:  # regions 1 and 3
                axs[t, 0].annotate(f"Region {i + 1}", (corner[0] + box_sizes[i] / 2, corner[1] + 1.1 * box_sizes[i]),
                                   ha='center', va='center')
            else:
                axs[t, 0].annotate(f"Region {i + 1}", (corner[0] + box_sizes[i] / 2, corner[1] - box_sizes[i] / 4),
                                   ha='center', va='center')
        # Plot bar chart comparing the number of swarms in each region
        regions = ['Region 1', 'Region 2', 'Region 3', 'Region 4']
        counts_x1 = x1_counts[t]
        counts_x2 = x2_counts[t]
        bar_width = 0.35

        x = np.arange(len(regions))
        axs[t, 1].bar(x, counts_x1, width=bar_width, label='Attackers', color='red', alpha=0.5)
        axs[t, 1].bar(x + bar_width, counts_x2, width=bar_width, label='Defenders', color='blue', alpha=0.5)

        axs[t, 1].set_xticks(x + bar_width / 2)
        axs[t, 1].set_xticklabels(regions)
        axs[t, 1].set_ylabel('Number of Swarms')
        axs[t, 1].set_title('Number of Swarms in Each Region')
        axs[t, 1].legend()
    plt.tight_layout()
    plt.show()

