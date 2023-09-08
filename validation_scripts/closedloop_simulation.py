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
    dvdt = jac[..., 0]
    dvdx = jac[..., 1:]

    # separate into v1 and v2
    dv1dx = dvdx[..., 0, :]
    dv2dx = dvdx[..., 1, :]

    # costate for attacker (P1)
    lam_11 = dv1dx[..., :1]
    lam_12 = dv1dx[..., 1:2]
    lam_13 = dv1dx[..., 2:3]
    lam_14 = dv1dx[..., 3:4]

    # costate for defender (P2)
    lam_21 = dv2dx[..., :1]
    lam_22 = dv2dx[..., 1:2]
    lam_23 = dv2dx[..., 2:3]
    lam_24 = dv2dx[..., 3:4]

    # for attacker (P1)
    u1 = torch.sign((lam_12 - lam_11))
    u2 = torch.sign((lam_11 - lam_12))

    u1[u1 < 0] = 0
    u2[u2 < 0] = 0

    # for defender (P2)
    d1 = torch.sign((lam_24 - lam_23))
    d2 = torch.sign((lam_23 - lam_24))

    d1[d1 < 0] = 0
    d2[d2 < 0] = 0

    return u1.item(), u2.item(), d1.item(), d2.item()


if __name__ == "__main__":
    nl = 'sine'
    model = modules.SingleBVPNet(in_features=5, out_features=2, type=nl, mode='mlp',
                                 final_layer_factor=1., hidden_features=32, num_hidden_layers=3)

    ckpt_path = '../experiment_scripts/logs/swarm_hji-2d_2.5_general_test/checkpoints/model_final.pth'

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


    ## simulate trajectory for one initial state (0.5, 0.5)

    data = scio.loadmat('data_train2.5s_general.mat')

    T = data['t']

    idxs = np.where(T == 0)[1]  # these are the start point of trajectories

    ts = [T.flatten()[idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
    ts.append(T.flatten()[idxs[-1]:])

    X = data['X']
    Xs = [X[:, idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
    Xs.append(X[:, idxs[-1]:])  # add the last remaining trajectory

    A = data['A']
    As = [A[:, idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
    As.append(A[:, idxs[-1]:])


    U = data['U']
    Us = [U[:, idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
    Us.append(U[:, idxs[-1]:])

    V = data['V']
    Vs = [V[:, idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
    Vs.append(V[:, idxs[-1]:])

    idx = 1

    t = ts[idx]
    Vs = Vs[idx]

    As = As[idx]

    X_1 = Xs[idx]  # pick 1 trajector and see
    X_1_1 = X_1[:2, :].T
    X_1_2 = X_1[2:, :].T

    U_1 = Us[idx]
    U_1_1 = U_1[:2, :].T
    U_1_2 = U_1[2:, :].T

    time_steps = t

    N = 11

    X1 = np.empty((2, N))
    X2 = np.empty((2, N))

    U = np.empty((2, N-1))
    D = np.empty((2, N-1))
    # Lam = np.empty((4, N-1))

    V = np.empty((1, N))

    # X = 0.5 * torch.ones(1, 4)
    # x1 = torch.zeros(1, 1).uniform_(0, 1)
    # x2 = torch.zeros(1, 1).uniform_(0, 1)
    # X = torch.cat((x1, 1-x1, x2, 1-x2), dim=1)

    # X = torch.from_numpy(X_1[:, 0].reshape(1, -1)).to(torch.float32)
    x_1 = np.concatenate((X_1_1[0, :], X_1_2[0, :])).reshape(1, -1)
    # x_2 = np.concatenate((X_1_2[0, :], X_1_1[0, :])).reshape(1, -1)

    # X = np.concatenate((x_1, x_2), axis=0)

    # X = torch.from_numpy(X).to(torch.float32)
    X = torch.from_numpy(x_1).to(torch.float32)

    X1[:, 0] = X[0, :2].numpy()
    X2[:, 0] = X[0, 2:].numpy()



    T = np.linspace(0, 2.5, N)
    T = np.flip(T)
    dt = 2.5/N


    for i in range(0, len(T)-1):
        time = T[i] * torch.ones(1, 1)
        # label = torch.zeros_like(time)
        # label[1, :] = 0
        # coords = torch.cat((time, X, label), dim=1).unsqueeze(0)
        coords = torch.cat((time, X), dim=1).unsqueeze(0)
        coords_in = {'coords': coords}
        value = model(coords_in)['model_out'].detach().cpu().numpy().squeeze()[0]
        u1, u2, d1, d2 = get_action_from_nn(coords_in, model)
        x1 = X[0, 0]
        x2 = X[0, 1]
        y1 = X[0, 2]
        y2 = X[0, 3]

        x1 += dt * (u2 * x2 - u1 * x1)
        x2 += dt * (u1 * x1 - u2 * x2)

        y1 += dt * (d2 * y2 - d1 * y1)
        y2 += dt * (d1 * y1 - d2 * y2)

        x_1 = torch.cat((x1.reshape(-1, 1), x2.reshape(-1, 1), y1.reshape(-1, 1), y2.reshape(-1, 1))).reshape(1, -1)
        # x_2 = torch.cat((y1.reshape(-1, 1), y2.reshape(-1, 1), x1.reshape(-1, 1), x2.reshape(-1, 1))).reshape(1, -1)

        X = x_1

        X1[:, i+1] = X[0, :2].detach().cpu().numpy()
        X2[:, i + 1] = X[0, 2:].detach().cpu().numpy()

        U[:, i] = np.concatenate(([u1], [u2]))
        D[:, i] = np.concatenate(([d1], [d2]))
        V[:, i] = value.item()
        # Lam[:, i] = lam

X = torch.cat((torch.tensor([0]).reshape(1, -1), X[0, :].reshape(1, -1)), dim=1)
V[:, N-1] = model({'coords': X.to(torch.float32)})['model_out'].detach().cpu().numpy()[0][0].item()



# plot states
X1 = np.vstack(X1).T
X2 = np.vstack(X2).T

# c_time = utils.check_target_status_2d(X1, X2, 0.8)

# idx = np.where(c_time == True)[0][0]

# capture_time = np.flip(T)[idx]

print("Attacker's Action: \n", U)
print("Defender's Action: \n", D)

print("Values: \n", V)

# print("Capture happens at t = ", capture_time)

fig, axs = plt.subplots(2)
axs[0].plot(np.flip(T), X1[:, 0], label='P1 at Region 1')
axs[0].plot(np.flip(T), X2[:, 0], label='P2 at Region 1')
axs[0].set_ylim([0, 1])
axs[0].legend()


axs[1].plot(np.flip(T), X1[:, 1], label='P1 at Region 2')
axs[1].plot(np.flip(T), X2[:, 1], label='P2 at Region 2')
axs[1].set_ylim([0, 1])
axs[1].legend()

plt.show()


# utils.plot_2d(X1, X2, np.flip(T))



# print("Final States: P1: ", X1[-1])
# print("Final States: P2: ", )







