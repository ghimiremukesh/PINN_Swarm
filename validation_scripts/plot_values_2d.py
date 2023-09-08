import scipy.io as scio
import numpy as np
import torch
import matplotlib.pyplot as plt
import modules
from scipy.interpolate import griddata

# first plot value from bvp solver
data = scio.loadmat("data_train2.5s_general.mat")

T = data['t']

idxs = np.where(T == 0)[1]  # these are the start point of trajectories

ts = [T.flatten()[idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
ts.append(T.flatten()[idxs[-1]:])

X = data['X']
Xs = [X[:, idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
Xs.append(X[:, idxs[-1]:])  # add the last remaining trajectory

U = data['U']
Us = [U[:, idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
Us.append(U[:, idxs[-1]:])

V = data['V']
Vs = [V[:, idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
Vs.append(V[:, idxs[-1]:])


X1_initial = np.array([Xs[i][0, 0] for i in range(len(Xs))])
X2_initial = np.array([Xs[i][2, 0] for i in range(len(Xs))])

V_initial = np.array([Vs[i][0, 0] for i in range(len(Vs))])


plt.scatter(X1_initial, X2_initial, c=V_initial, cmap='bwr_r', vmin=np.min(V_initial), vmax=np.max(V_initial))
plt.colorbar()
plt.show()


nl = 'relu'
model = modules.SingleBVPNet(in_features=6, out_features=1, type=nl, mode='mlp',
                             final_layer_factor=1., hidden_features=32, num_hidden_layers=3)

ckpt_path = '../experiment_scripts/logs/swarm_hji-2d_2.5_general/checkpoints/model_final.pth'

ckpt = torch.load(ckpt_path)

model.load_state_dict(ckpt)

model.eval()

num_points = 10000

# torch.manual_seed(10)

# x1 = torch.zeros(num_points, 1).uniform_(0, 1)
# x2 = torch.zeros(num_points, 1).uniform_(0, 1)

X1_initial = torch.from_numpy(X1_initial).to(torch.float32)
X2_initial = torch.from_numpy(X2_initial).to(torch.float32)

X = torch.cat((X1_initial.reshape(-1, 1), 1-X1_initial.reshape(-1, 1),
               X2_initial.reshape(-1, 1), 1-X2_initial.reshape(-1, 1)), dim=1)

# X = torch.cat((x1, 1-x1, x2, 1-x2), dim=1)

time = 2.5 * torch.ones(num_points, 1)

coords = torch.cat((time, X), dim=1)

coords = torch.cat((coords, torch.zeros_like(coords[:, 0]).reshape(-1, 1)), dim=1)  # after adding label

coords_in = {'coords': coords}

value = model(coords_in)['model_out'].detach().cpu().numpy()

# plt.scatter(x1, x2, c=value, cmap='bwr', vmin=-np.min(value), vmax=np.max(value))
# marker_size = 50 * (value - np.min(value)) / (np.max(value) - np.min(value)) + 10

# Create scatter plot with variable marker size and color
plt.scatter(X1_initial, X2_initial, c=value, cmap='bwr_r', vmin=np.min(V_initial), vmax=np.max(V_initial))



plt.colorbar()
plt.show()