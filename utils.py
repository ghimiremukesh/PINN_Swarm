import numpy as np
import torch
import os

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
    X0 = np.concatenate((x0_1, x0_2), axis=1) # concatenate into N x 8

    return X0.astype(np.float32)


def softmax(x, alpha):
    return np.exp(alpha * x)/sum(np.exp(alpha * x))


def boltzmann_operator(x, alpha):
    return sum(x * np.exp(alpha * x))/sum(np.exp(alpha * x))