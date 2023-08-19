import torch
import diff_operators
import os
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def initialize_swarm_hji(dataset):
    def swarm_hji(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
        y = model_output['model_out']
        dirichlet_mask = gt['dirichlet_mask']

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)

        # partial gradient w.r.t time and state
        dvdt = jac[..., 0, 0].squeeze()
        dvdx = jac[..., 0, 1:].squeeze()


        # costate for attacker (P1)
        lam_11 = dvdx[:, :1]
        lam_12 = dvdx[:, 1:2]
        lam_13 = dvdx[:, 2:3]
        lam_14 = dvdx[:, 3:4]

        # costate for defender (P2)
        lam_21 = dvdx[:, 4:5]
        lam_22 = dvdx[:, 5:6]
        lam_23 = dvdx[:, 6:7]
        lam_24 = dvdx[:, 7:8]

        # since hamiltonian is decoupled, we can apply min max separately
        # for attacker (P1)
        u1 = dataset.u_max * torch.sign(x[..., 1].T * (lam_12 - lam_11))
        u2 = dataset.u_max * torch.sign(x[..., 2].T * (lam_13 - lam_12))
        u3 = dataset.u_max * torch.sign(x[..., 3].T * (lam_14 - lam_13))
        u4 = dataset.u_max * torch.sign(x[..., 4].T * (lam_11 - lam_14))


        u1[u1 <= 0] = dataset.u_min
        u2[u2 <= 0] = dataset.u_min
        u3[u3 <= 0] = dataset.u_min
        u4[u4 <= 0] = dataset.u_min

        # for defender (P2)
        d1 = dataset.u_max * torch.sign(x[..., 5].T * (lam_22 - lam_21))
        d2 = dataset.u_max * torch.sign(x[..., 6].T * (lam_23 - lam_22))
        d3 = dataset.u_max * torch.sign(x[..., 7].T * (lam_24 - lam_23))
        d4 = dataset.u_max * torch.sign(x[..., 8].T * (lam_21 - lam_24))

        d1[d1 <= 0] = dataset.u_min
        d2[d2 <= 0] = dataset.u_min
        d3[d3 <= 0] = dataset.u_min
        d4[d4 <= 0] = dataset.u_min

        # calculate hamiltonian
        lam_1 = torch.cat((lam_11, lam_12, lam_13, lam_14), dim=1)
        f1 = torch.cat((u4 * x[..., 4].reshape(-1, 1) - u1 * x[..., 1].reshape(-1, 1),
                           u1 * x[..., 1].reshape(-1, 1) - u2 * x[..., 2].reshape(-1, 1),
                           u2 * x[..., 2].reshape(-1, 1) - u3 * x[..., 3].reshape(-1, 1),
                           u3 * x[..., 3].reshape(-1, 1) - u4 * x[..., 4].reshape(-1, 1)), dim=1).T

        lam_2 = torch.cat((lam_21, lam_22, lam_23, lam_24), dim=1)
        f2 = torch.cat((d4 * x[..., 8].reshape(-1, 1) - d1 * x[..., 5].reshape(-1, 1),
                           d1 * x[..., 5].reshape(-1, 1) - d2 * x[..., 6].reshape(-1, 1),
                           d2 * x[..., 6].reshape(-1, 1) - d3 * x[..., 7].reshape(-1, 1),
                           d3 * x[..., 7].reshape(-1, 1) - d4 * x[..., 8].reshape(-1, 1)), dim=1).T

        ham1 = torch.sum(torch.mul(lam_1, f1.t()), dim=1) #torch.diag(lam_1 @ f1)
        ham2 = torch.sum(torch.mul(lam_2, f2.t()), dim=1) # torch.diag(lam_2 @ f2)

        # ham1 = np.sum(np.multiply(lam_1.detach().numpy(), f1.t().detach().numpy()), axis=1) #torch.diag(lam_1 @ f1)
        # ham2 = np.sum(np.multiply(lam_2.detach().numpy(), f2.t().detach().numpy()), axis=1) # torch.diag(lam_2 @ f2)

        ham = ham1 + ham2


        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            diff_constraint_hom = dvdt + ham

        # boundary condition check
        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        return {'dirichlet': torch.abs(dirichlet).sum(),
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return swarm_hji


def initialize_swarm_hji_2d(dataset):
    def swarm_hji(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
        y = model_output['model_out']
        dirichlet_mask = gt['dirichlet_mask']

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)

        # partial gradient w.r.t time and state
        dvdt = jac[..., 0, 0].squeeze()
        dvdx = jac[..., 0, 1:].squeeze()


        # costate for attacker (P1)
        lam_11 = dvdx[:, :1]
        lam_12 = dvdx[:, 1:2]

        # costate for defender (P2)
        lam_21 = dvdx[:, 4:5]
        lam_22 = dvdx[:, 5:6]


        # since hamiltonian is decoupled, we can apply min max separately
        # for attacker (P1)
        u1 = dataset.u_max * torch.sign(x[..., 1].T * (lam_12 - lam_11))
        u2 = dataset.u_max * torch.sign(x[..., 2].T * (lam_11 - lam_12))

        u1[u1 <= 0] = dataset.u_min
        u2[u2 <= 0] = dataset.u_min


        # for defender (P2)
        d1 = dataset.u_max * torch.sign(x[..., 3].T * (lam_22 - lam_21))
        d2 = dataset.u_max * torch.sign(x[..., 4].T * (lam_21 - lam_22))

        d1[d1 <= 0] = dataset.u_min
        d2[d2 <= 0] = dataset.u_min


        # calculate hamiltonian
        lam_1 = torch.cat((lam_11, lam_12), dim=1)
        f1 = torch.cat((u2 * x[..., 2].reshape(-1, 1) - u1 * x[..., 1].reshape(-1, 1),
                           u1 * x[..., 1].reshape(-1, 1) - u2 * x[..., 2].reshape(-1, 1)), dim=1).T

        lam_2 = torch.cat((lam_21, lam_22), dim=1)
        f2 = torch.cat((d2 * x[..., 4].reshape(-1, 1) - d1 * x[..., 3].reshape(-1, 1),
                           d1 * x[..., 3].reshape(-1, 1) - d2 * x[..., 4].reshape(-1, 1)), dim=1).T

        ham1 = torch.sum(torch.mul(lam_1, f1.t()), dim=1) #torch.diag(lam_1 @ f1)
        ham2 = torch.sum(torch.mul(lam_2, f2.t()), dim=1) # torch.diag(lam_2 @ f2)

        # ham1 = np.sum(np.multiply(lam_1.detach().numpy(), f1.t().detach().numpy()), axis=1) #torch.diag(lam_1 @ f1)
        # ham2 = np.sum(np.multiply(lam_2.detach().numpy(), f2.t().detach().numpy()), axis=1) # torch.diag(lam_2 @ f2)

        ham = ham1 + ham2


        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            diff_constraint_hom = dvdt + ham

        # boundary condition check
        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        return {'dirichlet': torch.abs(dirichlet).sum(),
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return swarm_hji