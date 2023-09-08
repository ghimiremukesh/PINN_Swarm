import torch
from torch.utils.data import Dataset
import scipy.io
import os
import math

import utils


class SwarmHJI(Dataset):
    def __init__(self, numpoints, alpha, u_max=1, u_min=0, t_min=0, t_max=1, counter_start=0, counter_end=100e3,
                 pretrain=True, pretrain_iters=10000, num_src_samples=1000, seed=0):
        super().__init__()
        torch.manual_seed(seed)

        self.numpoints = numpoints
        self.u_min = u_min
        self.u_max = u_max

        self.t_min = t_min
        self.t_max = t_max

        self.alpha = alpha  # temperature parameter

        self.num_states = 4  # each agent's state is defined by 4 elements

        self.N_src_samples = num_src_samples

        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.

        # sample such that each agent's states sum to 1
        states = torch.from_numpy(utils.sample_X0(self.numpoints, self.num_states))  # returns N x 8 matrix

        if self.pretrain:
            # only sample in time around initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, states), dim=1)
        else:
            # slowly grow time
            time = self.t_min + torch.zeros(self.numpoints, 1).uniform_(0, (self.t_max - self.t_min) *
                                                                        (self.counter / self.full_count))
            coords = torch.cat((time, states), dim=1)

            # make sure we have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # boundary values
        X1_T = coords[:, 1:self.num_states + 1]
        X2_T = coords[:, self.num_states + 1:]

        del_XT = X1_T - X2_T

        # boundary_values = torch.stack([utils.boltzmann_operator(row, self.alpha) for row in del_XT]).reshape(-1, 1)

        boundary_values = utils.boltzmann_vec(del_XT, self.alpha).reshape(-1, 1)

        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        return {'coords': coords}, {'source_boundary_values': boundary_values,
                                    'dirichlet_mask': dirichlet_mask}


class SwarmHJI_2d(Dataset):
    def __init__(self, numpoints, alpha, u_max=1, u_min=0, t_min=0, t_max=1, counter_start=0, counter_end=100e3,
                 pretrain=True, pretrain_iters=10000, num_src_samples=1000, seed=0):
        super().__init__()
        torch.manual_seed(seed)

        self.numpoints = numpoints
        self.u_min = u_min
        self.u_max = u_max

        self.t_min = t_min
        self.t_max = t_max

        self.alpha = alpha  # temperature parameter

        self.num_states = 2  # each agent's state is defined by 4 elements

        self.N_src_samples = num_src_samples

        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.

        # sample such that each agent's states sum to 1
        # contains (x1, x2) and (x2, x1) pairs
        states = torch.from_numpy(utils.sample_X0(self.numpoints, self.num_states))  # returns N x 8 matrix

        if self.pretrain:
            # only sample in time around initial condition
            time = torch.ones(self.numpoints * 2, 1) * start_time
            coords = torch.cat((time, states), dim=1)
        else:
            # slowly grow time
            time = self.t_min + torch.zeros(self.numpoints * 2, 1).uniform_(0, (self.t_max - self.t_min) *
                                                                        (self.counter / self.full_count))
            coords = torch.cat((time, states), dim=1)

            # make sure we have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time
            coords[-(self.numpoints+self.N_src_samples):-self.numpoints, 0] = start_time

        # boundary values
        coords = torch.cat((coords, torch.ones_like(coords[:, 0]).reshape(-1, 1)), dim=1)  # label for v2
        coords[:self.numpoints, -1] = 0  # label for v1
        X1_T = coords[:, 1:self.num_states + 1]
        X2_T = coords[:, self.num_states + 1:-1]

        del_XT = X1_T - X2_T

        # boundary_values = torch.stack([utils.boltzmann_operator(row, self.alpha) for row in del_XT]).reshape(-1, 1)
        boundary_values = utils.boltzmann_vec(del_XT, self.alpha).reshape(-1, 1)

        # # add boundary costate values
        # boundary_costates = torch.mul(utils.softmax(del_XT, self.alpha),
        #                     (1 + self.alpha * (del_XT - boundary_values)))
        #
        # boundary_costates = torch.cat((boundary_costates, -boundary_costates), dim=0)

        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        # return {'coords': coords}, {'source_boundary_values': boundary_values,
        #                             'source_boundary_costates': boundary_costates,
        #                             'dirichlet_mask': dirichlet_mask}

        return {'coords': coords}, {'source_boundary_values': boundary_values,
                                    'dirichlet_mask': dirichlet_mask}


class SwarmHJI_2d_new(Dataset):
    def __init__(self, numpoints, alpha, u_max=1, u_min=0, t_min=0, t_max=1, counter_start=0, counter_end=100e3,
                 pretrain=True, pretrain_iters=10000, num_src_samples=1000, seed=0):
        super().__init__()
        torch.manual_seed(seed)

        self.numpoints = numpoints
        self.u_min = u_min
        self.u_max = u_max

        self.t_min = t_min
        self.t_max = t_max

        self.alpha = alpha  # temperature parameter

        self.num_states = 2  # each agent's state is defined by 4 elements

        self.N_src_samples = num_src_samples

        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.

        # sample such that each agent's states sum to 1
        # contains (x1, x2) and (x2, x1) pairs
        states = torch.from_numpy(utils.sample_X0_uniform(self.numpoints, self.num_states))  # returns N x 8 matrix

        if self.pretrain:
            # only sample in time around initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, states), dim=1)
        else:
            # slowly grow time
            time = self.t_min + torch.zeros(self.numpoints, 1).uniform_(0, (self.t_max - self.t_min) *
                                                                        (self.counter / self.full_count))
            coords = torch.cat((time, states), dim=1)

            # make sure we have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # boundary values
        X1_T = coords[:, 1:self.num_states + 1]
        X2_T = coords[:, self.num_states + 1:]

        del_XT = X1_T - X2_T
        del_XT_2 = X2_T - X1_T

        # boundary_values = torch.stack([utils.boltzmann_operator(row, self.alpha) for row in del_XT]).reshape(-1, 1)
        boundary_values_1 = utils.boltzmann_vec(del_XT, self.alpha).reshape(-1, 1)
        boundary_values_2 = utils.boltzmann_vec(del_XT_2, self.alpha).reshape(-1, 1)

        boundary_values = torch.cat((boundary_values_1, boundary_values_2), dim=1)


        # add boundary costate values
        # boundary_grads_1 = torch.mul(utils.softmax(del_XT, self.alpha),
        #                     (1 + self.alpha * (del_XT - boundary_values)))
        # boundary_grads_2 = torch.mul(utils.softmax(del_XT_2, self.alpha),
        #                     (1 + self.alpha * (del_XT - boundary_values)))
        # #
        # boundary_grads_1 = torch.cat((boundary_grads_1, -boundary_grads_1), dim=1)
        # boundary_grads_2 = torch.cat((-boundary_grads_2, boundary_grads_2), dim=1)
        #
        # boundary_grads = torch.cat((boundary_grads_1, boundary_grads_2), dim=1).unsqueeze(0)


        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        dirichlet_mask = dirichlet_mask.expand_as(boundary_values)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        return {'coords': coords}, {'source_boundary_values': boundary_values,
                                    'dirichlet_mask': dirichlet_mask}

        # return {'coords': coords}, {'source_boundary_values': boundary_values,
        #                             'source_boundary_grads': boundary_grads,
        #                             'dirichlet_mask': dirichlet_mask}