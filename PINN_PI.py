import sys
# update your projecty root path before running
# sys.path.insert(0, '/path/to/nsga-net')
# sys.path.append(r'/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2')
# sys.path.append(r'/home/weiz/Software/Software/anaconda3/pkgs/mkl-2023.1.0-h6d00ec8_46342/lib')


import os
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
# import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd.functional import jacobian
from torch.utils.data import DataLoader, TensorDataset

import time

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
from DNN import DNN_PI
from thop import profile

# device = 'cuda'

# Bicycle model parameters
L = 2.5  # wheelbase

def bicycle_dynamics(state, control):
    if state.ndim == 1:
        state = state.unsqueeze(0)
        control = control.unsqueeze(0)

    x = state[:, 0]
    y = state[:, 1]
    theta = state[:, 2]
    v = state[:, 3]
    a = control[:, 0]
    delta = control[:, 1]

    dx = v * torch.cos(theta)
    dy = v * torch.sin(theta)
    dtheta = v / L * torch.tan(delta)
    dv = a

    return torch.stack([dx, dy, dtheta, dv], dim=1)

def rk4_step(state, control, dt):
    if state.ndim == 1:
        state = state.unsqueeze(0)
        control = control.unsqueeze(0)

    k1 = bicycle_dynamics(state, control)
    k2 = bicycle_dynamics(state + 0.5 * dt * k1, control)
    k3 = bicycle_dynamics(state + 0.5 * dt * k2, control)
    k4 = bicycle_dynamics(state + dt * k3, control)
    return (state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)).squeeze(0)

def simulate_interval_torch(x0, u, T, dt):
    times = torch.arange(0, T + dt, dt, device=x0.device)
    states = [x0]
    x = x0.clone()
    for _ in range(1, len(times)):
        x = rk4_step(x, u, dt)
        states.append(x)
    return times, torch.stack(states)

def generate_training_batch(batch_size=64, T=1.0, dt=0.05):
    N = int(T / dt) + 1
    t_vals = torch.linspace(0, T, N).unsqueeze(1).repeat(1, batch_size).T  # (B, N)
    x0_batch = torch.rand(batch_size, 4) * torch.tensor([5.0, 5.0, 2*np.pi, 2.0]) - torch.tensor([0.0, 0.0, np.pi, 0.0])
    u_batch = torch.rand(batch_size, 2) * torch.tensor([2.0, 1.0]) - torch.tensor([1.0, 0.5])

    t_vals = t_vals.reshape(-1, 1)
    inputs = []
    targets = []
    for i in range(batch_size):
        t_i, x_seq = simulate_interval_torch(x0_batch[i], u_batch[i], T, dt)
        x0_repeat = x0_batch[i].repeat(len(t_i), 1)
        u_repeat = u_batch[i].repeat(len(t_i), 1)
        input_i = torch.cat([t_i.unsqueeze(1), x0_repeat, u_repeat], dim=1)
        inputs.append(input_i)
        targets.append(x_seq)

    return torch.cat(inputs, dim=0), torch.cat(targets, dim=0)

batch_size = 50
T = 5.0
dt = 0.05
x_train, y_train = generate_training_batch(batch_size, T, dt)
x_train_all, y_train_all = x_train.detach().cpu().numpy(), y_train.detach().cpu().numpy()
_ic = x_train_all[:, 0] == 0
x_train_ic = x_train_all[_ic]
y_train_ic = y_train_all[_ic]


def PINN_PI(seed=0, gpu=0):

    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)

    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    input_dim = 7
    n_nodes_first_layer = 512
    n_nodes_list = [256, 256, 4]
    i_ac_list = [9, 9]


    model = DNN_PI(input_dim, n_nodes_first_layer, n_nodes_list, i_ac_list).to(device)

    checkpoint = torch.load('trained_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    mse_all, mse_list = infer(model, x_train_all, y_train_all, x_train_ic, y_train_ic, n_nodes_list, device)
        
    # # save model
    # torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'MSE': mse}, 'train_model.pt')

    return mse_all, mse_list

# Infer
def infer(model, x_train_all, y_train_all, x_train_ic, y_train_ic, n_nodes_list, device):
    model.eval()
    flag = 1
    # x_train_all = torch.tensor(x_train_all).to(device).float()
    # y_train_all = torch.tensor(y_train_all).to(device).float()
    # x_train_ic = torch.tensor(x_train_ic).to(device).float()
    y_train_ic = torch.tensor(y_train_ic).to(device).float()
    mse_list = []
    mse_all = 0

    for i_b in np.arange(batch_size):
        t = torch.tensor(x_train_all[i_b*(int(T/dt) + 1):(i_b + 1)*(int(T/dt) + 1), 0:1], requires_grad=True).to(device).float()
        x0 = torch.tensor(x_train_all[i_b*(int(T/dt) + 1):(i_b + 1)*(int(T/dt) + 1), 1:5], requires_grad=True).to(device).float()
        u_control = torch.tensor(x_train_all[i_b*(int(T/dt) + 1):(i_b + 1)*(int(T/dt) + 1), 5:7], requires_grad=True).to(device).float()
        label = torch.tensor(y_train_all[i_b*(int(T/dt) + 1):(i_b + 1)*(int(T/dt) + 1), :]).to(device).float()

        f = model(torch.cat([t, x0, u_control], dim=1), flag)

        f_t_list = []
        for i in range(f.shape[1]):
            grad_i = torch.autograd.grad(
                f[:, i:i+1], t,
                grad_outputs=torch.ones_like(f[:, i:i+1]),
                retain_graph=True,
                create_graph=True
            )[0]
            f_t_list.append(grad_i)
        f_t = torch.cat(f_t_list, dim=1)

        n_e = 4
        n_s = t.shape[0]
        n_nodes = n_nodes_list[-2]
        L = 2.5  # wheelbase
        a = u_control[:, 0:1]
        delta = u_control[:, 1:2]

        # IC
        ic_A1 = torch.zeros((1, n_e*n_nodes)).to(device)
        ic_A1[:, 0*n_nodes:1*n_nodes] = f[0:1]

        ic_A2 = torch.zeros((1, n_e*n_nodes)).to(device)
        ic_A2[:, 1*n_nodes:2*n_nodes] = f[0:1]

        ic_A3 = torch.zeros((1, n_e*n_nodes)).to(device)
        ic_A3[:, 2*n_nodes:3*n_nodes] = f[0:1]

        ic_A4 = torch.zeros((1, n_e*n_nodes)).to(device)
        ic_A4[:, 3*n_nodes:4*n_nodes] = f[0:1]

        ic_A = torch.vstack([ic_A1, ic_A2, ic_A3, ic_A4])
        ic_b = y_train_ic[i_b].reshape(-1, 1)

        initial_guess_value = torch.tensor(0)
        for i in range(6):
            # PDE (nonlinear)
            if i == 0:
                theta = initial_guess_value
            else:
                theta = f @ w_PI[2*n_nodes:3*n_nodes, :]

            pde_A1 = torch.zeros((n_s, n_e*n_nodes)).to(device)
            pde_A1[:, 0*n_nodes:1*n_nodes] = f_t
            pde_A1[:, 3*n_nodes:4*n_nodes] = -f * torch.cos(theta)

            pde_A2 = torch.zeros((n_s, n_e*n_nodes)).to(device)
            pde_A2[:, 1*n_nodes:2*n_nodes] = f_t
            pde_A2[:, 3*n_nodes:4*n_nodes] = -f * torch.sin(theta)

            pde_A3 = torch.zeros((n_s, n_e*n_nodes)).to(device)
            pde_A3[:, 2*n_nodes:3*n_nodes] = f_t
            pde_A3[:, 3*n_nodes:4*n_nodes] = -f / L * torch.tan(delta)

            pde_A4 = torch.zeros((n_s, n_e*n_nodes)).to(device)
            pde_A4[:, 3*n_nodes:4*n_nodes] = f_t

            pde_A = torch.vstack([pde_A1, pde_A2, pde_A3, pde_A4])

            pde_b = torch.zeros((pde_A.shape[0], 1)).to(device)
            pde_b[3*n_s:4*n_s, :] = a

            A = torch.vstack([ic_A, pde_A])
            b = torch.vstack([ic_b, pde_b])

            reg = 1e-6

            A = A.detach().cpu().numpy()
            b = b.detach().cpu().numpy()
            w_PI = np.linalg.inv(reg*np.eye(A.shape[1]) + (A.T @ A)) @ A.T @ b
            w_PI = torch.tensor(w_PI).float().to(device)
            
        pred1 = f @ w_PI[0*n_nodes:1*n_nodes, :]
        pred2 = f @ w_PI[1*n_nodes:2*n_nodes, :]
        pred3 = f @ w_PI[2*n_nodes:3*n_nodes, :]
        pred4 = f @ w_PI[3*n_nodes:4*n_nodes, :]
        pred_all = torch.hstack([pred1, pred2, pred3, pred4])
        
        # mse
        mse = torch.mean((label - pred_all) ** 2)
        mse_list.append(mse)
        mse_all = mse_all + mse
    
    mse_all = mse_all / batch_size
    return mse_all, mse_list

if __name__ == "__main__":
    gpu = 0
    loss_min, mse_min = PINN_PI(seed=0, gpu=0)
    print(f"Test MSE: {loss_min.item()}, mse_list: {mse_min}")
