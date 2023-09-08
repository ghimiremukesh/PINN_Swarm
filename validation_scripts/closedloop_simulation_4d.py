import sys
import os

import utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules, diff_operators
import time
import torch
import numpy as np
import scipy.io as scio

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# use cpu for evaluation
device = torch.device("cpu")


def dynamics(X, dt, action):
    u, d = action
    u1, u2 = u
    d1, d2 = d

    X1 = X[:, :2]
    X2 = X[:, 2:]

    # player 1
    x1 = X1[:, 0]
    x2 = X1[:, 1]

    # player 2
    y1 = X2[:, 0]
    y2 = X2[:, 1]

    dx1 = u2 * x2 - u1 * x1
    dx2 = u1 * x1 - u2 * x2

    dy1 = d2 * y2 - d1 * y1
    dy2 = d1 * y1 - d2 * y2

    x1_prime = x1 + dx1 * dt
    x2_prime = x2 + dx2 * dt
    y1_prime = y1 + dy1 * dt
    y2_prime = y2 + dy2 * dt

    X_next = np.concatenate((x1_prime, x2_prime, y1_prime, y2_prime), axis=1)

    return X_next


def get_action_from_nn(coords_in, model):
    model_output = model(coords_in)
    x = model_output['model_in']
    y = model_output['model_out']

    # calculate the partial gradient of V w.r.t. time and state
    jac, _ = diff_operators.jacobian(y, x)

    # partial gradient w.r.t time and state
    dvdt = jac[..., 0, 0].squeeze()
    dvdx = jac[..., 0, 1:].squeeze()

    # costate for attacker (P1)
    lam_11 = dvdx[:1]
    lam_12 = dvdx[1:2]
    lam_13 = dvdx[2:3]
    lam_14 = dvdx[3:4]

    # costate for defender (P2)
    lam_21 = dvdx[4:5]
    lam_22 = dvdx[5:6]
    lam_23 = dvdx[6:7]
    lam_24 = dvdx[7:8]

    # since hamiltonian is decoupled, we can apply min max separately
    # for attacker (P1)
    u1 = 1 * torch.sign(x[..., 1] * (lam_12 - lam_11))
    u2 = 1 * torch.sign(x[..., 2] * (lam_13 - lam_12))
    u3 = 1 * torch.sign(x[..., 3] * (lam_14 - lam_13))
    u4 = 1 * torch.sign(x[..., 4] * (lam_11 - lam_14))

    u1[u1 <= 0] = 0
    u2[u2 <= 0] = 0
    u3[u3 <= 0] = 0
    u4[u4 <= 0] = 0

    # for defender (P2)
    d1 = 1 * torch.sign(x[..., 5] * (lam_22 - lam_21))
    d2 = 1 * torch.sign(x[..., 6] * (lam_23 - lam_22))
    d3 = 1 * torch.sign(x[..., 7] * (lam_24 - lam_23))
    d4 = 1 * torch.sign(x[..., 8] * (lam_21 - lam_24))

    d1[d1 >= 0] = 0
    d2[d2 >= 0] = 0
    d3[d3 >= 0] = 0
    d4[d4 >= 0] = 0

    d1[d1 < 0] = 1
    d2[d2 < 0] = 1
    d3[d3 < 0] = 1
    d4[d4 < 0] = 1

    return u1, u2, u3, u4, d1, d2, d3, d4


if __name__ == "__main__":
    model = modules.SingleBVPNet(in_features=9, out_features=1, type='relu', mode='mlp',
                                 final_layer_factor=1., hidden_features=32, num_hidden_layers=3)

    ckpt_path = '../experiment_scripts/logs/swarm_hji/checkpoints/model_final.pth'

    ckpt = torch.load(ckpt_path)

    model.load_state_dict(ckpt)

    model.eval()

    # num_points = 100000
    #
    # # torch.manual_seed(10)
    #
    # x1 = torch.zeros(num_points, 1).uniform_(0, 1)
    # x2 = torch.zeros(num_points, 1).uniform_(0, 1)
    #
    # X = torch.cat((x1, 1-x1, x2, 1-x2), dim=1)
    #
    # time = torch.zeros(num_points, 1)
    #
    # coords = torch.cat((time, X), dim=1)
    #
    # coords_in = {'coords': coords}
    #
    # value = model(coords_in)['model_out'].detach().cpu().numpy()
    #
    # plt.scatter(x1, x2, c=value, cmap='bwr_r', vmin=-0.1, vmax=0.4)
    # plt.colorbar()
    # plt.show()


    ## simulate trajectory for one initial state from the training data

    data = scio.loadmat("data_train2.5s_4d.mat")

    T = data['t']

    idxs = np.where(T == 0)[1]  # these are the start point of trajectories

    ts = [T.flatten()[idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
    ts.append(T.flatten()[idxs[-1]:])

    X = data['X']
    Xs = [X[:, idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
    Xs.append(X[:, idxs[-1]:])  # add the last remaining trajectory


    U = data['U']
    Us = [U[:, idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]

    V = data['V']
    Vs = [V[:, idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]

    idx = 10

    t = ts[idx]
    Vs = Vs[idx]

    X_1 = Xs[idx]  # pick 1 trajector and see
    X_1_1 = X_1[:4, :].T
    X_1_2 = X_1[4:, :].T

    U_1 = Us[idx]
    U_1_1 = U_1[:4, :].T
    U_1_2 = U_1[4:, :].T

    time_steps = t



    N = 11

    X1 = np.empty((4, N))
    X2 = np.empty((4, N))

    U = np.empty((4, N-1))
    D = np.empty((4, N-1))

    V = np.empty((1, N))

    # # X = 0.5 * torch.ones(1, 4)
    # x1 = torch.zeros(1, 4).uniform_(0, 1)
    # x1 = x1/x1.sum()
    # x2 = torch.zeros(1, 4).uniform_(0, 1)
    # x2 = x2/x2.sum()
    # X = torch.cat((x1, x2), dim=1)
    X = torch.from_numpy(X_1[:, 0].reshape(1, -1)).to(torch.float32)
    X1[:, 0] = X[:, :4].numpy()
    X2[:, 0] = X[:, 4:].numpy()



    T = np.linspace(0, 2.5, N)
    T = np.flip(T)
    dt = 2.5/N


    for i in range(0, len(T)-1):
        time = T[i] * torch.ones(1, 1)
        coords = torch.cat((time, X), dim=1)
        coords_in = {'coords': coords}
        u1, u2, u3, u4, d1, d2, d3, d4 = get_action_from_nn(coords_in, model)
        value = model(coords_in)['model_out'].detach().cpu().numpy().item()
        V[:, i] = value
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]

        y1 = X[:, 4]
        y2 = X[:, 5]
        y3 = X[:, 6]
        y4 = X[:, 7]

        x1 += dt * (u4 * x4 - u1 * x1)
        x2 += dt * (u1 * x1 - u2 * x2)
        x3 += dt * (u2 * x2 - u3 * x3)
        x4 += dt * (u3 * x3 - u4 * x4)

        y1 += dt * (d4 * y4 - d1 * y1)
        y2 += dt * (d1 * y1 - d2 * y2)
        y3 += dt * (d2 * y2 - d3 * y3)
        y4 += dt * (d3 * y3 - d4 * y4)

        X = torch.cat((x1, x2, x3, x4, y1, y2, y3, y4)).reshape(1, -1)

        X1[:, i+1] = X[:, :4].detach().cpu().numpy()
        X2[:, i + 1] = X[:, 4:].detach().cpu().numpy()

        U[:, i] = torch.cat((u1, u2, u3, u4)).detach().cpu().numpy()
        D[:, i] = torch.cat((d1, d2, d3, d4)).detach().cpu().numpy()

# X = torch.cat(())

# plot states
X1 = np.vstack(X1).T
X2 = np.vstack(X2).T

# c_time = utils.check_target_status_2d(X1, X2, 0.8)

# idx = np.where(c_time == True)[0][0]

# capture_time = np.flip(T)[idx]

print("Attacker's Action: \n", U.T)
print("Defender's Action: \n", D.T)

print("Values: \n", V)

# print('Costates: \n' )

# print("Capture happens at t = ", capture_time)




utils.plot_4d(X1, X2, T)

print("Final States: P1: ", X1[-1])
print("Final States: P2: ", )







