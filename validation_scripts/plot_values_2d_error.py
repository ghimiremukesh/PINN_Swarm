import scipy.io as scio
import numpy as np
import torch
import matplotlib.pyplot as plt

import diff_operators
import modules
from scipy.interpolate import griddata

plt.rcParams['font.size'] = 18

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

V2 = data['V2']
V2s = [V2[:, idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
V2s.append(V2[:, idxs[-1]:])

A = data['A']
As = [A[:, idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
As.append(A[:, idxs[-1]:])

nl = 'sine'
model = modules.SingleBVPNet(in_features=5, out_features=2, type=nl, mode='mlp',
                             final_layer_factor=1., hidden_features=32, num_hidden_layers=3)

ckpt_path = '../experiment_scripts/logs/swarm_hji-2d_2.5_general_test/checkpoints/model_final.pth'

ckpt = torch.load(ckpt_path)

model.load_state_dict(ckpt)

model.eval()

diffs = []

diffs_costates = []

total = 500

c_model = []
for i in range(total):
    X_curr = Xs[i].T
    t_curr = np.flip(ts[i]).reshape(-1, 1)
    val_curr_1 = Vs[i].squeeze()
    val_curr_2 = V2s[i].squeeze()

    cs = As[i].T
    costates_curr = cs.reshape(-1, 2, 4)
    # X_in = np.concatenate((t_curr, X_curr, np.zeros_like(t_curr)), axis=1)
    X_in = np.concatenate((t_curr, X_curr), axis=1)
    X_in = torch.from_numpy(X_in).to(torch.float32)
    coords_in = {'coords': X_in.unsqueeze(0)}
    model_out = model(coords_in)
    model_val_1 = model_out['model_out']
    coords_out = model_out['model_in']
    grad, _ = diff_operators.jacobian(model_val_1, coords_out)

    costates,  = grad[..., 1:].detach().cpu().numpy()

    model_v1 = model_val_1.detach().cpu().numpy().squeeze()[:, 0]
    model_v2 = model_val_1.detach().cpu().numpy().squeeze()[:, 1]

    diff = np.abs(model_v1 - val_curr_1) + np.abs(model_v2 - val_curr_2)

    diff_costates = np.abs(costates - costates_curr)

    diff_costates = np.apply_over_axes(np.sum, diff_costates, [1, 2]).squeeze()

    diffs_costates.append(diff_costates)

    diffs.append(diff)

    if i == 0:
        c_model.append(costates)




# keep trajectory of same length
T_adj = []
d_adj = []
c_adj = []
n = 11
for i in range(total):
    if len(diffs[i]) == n:
        d_adj.append(diffs[i])
        T_adj.append(ts[i])
        c_adj.append(diffs_costates[i])

c_adj = np.array(c_adj)
d_adj = np.array(d_adj)
mean_errors = np.mean(d_adj, axis=0)
var_errors = np.var(d_adj, axis=0)

mean_c_errors = np.mean(c_adj, axis=0)
var_c_errors = np.var(c_adj, axis=0)


plt.figure(figsize=(10, 8))
plt.plot(T_adj[0], mean_errors, label="Mean Absolute Error", color='blue')

plt.fill_between(T_adj[0], mean_errors - np.sqrt(var_errors), mean_errors + np.sqrt(var_errors),
                 alpha=0.2, color='blue')
plt.xlabel("Time-steps")
plt.ylabel("MAE")
plt.title("MAE of Values Between BVP Solution and NN Prediction")

# plt.savefig("Error_plot.png", dpi=300, bbox_inches=None)
plt.show()
# print(len(d_adj))

plt.figure(figsize=(10, 8))
plt.plot(T_adj[0], mean_c_errors, label="Mean Absolute Error Costates", color='blue')

plt.fill_between(T_adj[0], mean_c_errors - np.sqrt(var_c_errors), mean_c_errors + np.sqrt(var_c_errors),
                 alpha=0.2, color='blue')
plt.xlabel("Time-steps")
plt.ylabel("MAE")
plt.title("MAE of Costates Between BVP Solution and NN Prediction")

plt.show()

# plot one costate

c_model = c_model[0].reshape(-1, 8)

for n in range(8):
    c = As[0][n, :]
    cm = c_model[:, n]
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(c)
    axs[1].plot(cm)

plt.show()
#
# # plot diffs along time for all trajectories
#
# fig, ax = plt.subplots(figsize=(8, 8))
#
# for i in range(total):
#     ax.plot(ts[i], diffs[i])
#     ax.set_yscale('log')
#     ax.set_ylim([0, 0.1])
#
#
# plt.show()
#
# print()