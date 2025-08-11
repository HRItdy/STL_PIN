import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange


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

# Simple PINC-style neural network
class PINN(nn.Module):
    def __init__(self, input_dim=7, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim)
        )

    def forward(self, t, x0, u):
        inp = torch.cat([t, x0, u], dim=-1)
        return self.net(inp)

def physics_residual(model, t, x0, u, dt):
    t.requires_grad_(True)
    x_pred = model(t, x0, u)
    grads = torch.autograd.grad(x_pred, t, grad_outputs=torch.ones_like(x_pred), create_graph=True)[0]
    x_dot = bicycle_dynamics(x_pred, u)
    return grads - x_dot

def train_pinn(epochs=10000, batch_size=64, T=1.0, dt=0.05, lr=1e-3):
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in trange(epochs, desc="Training"): 
        x_train, y_train = generate_training_batch(batch_size, T, dt)
        x_train = x_train.detach()
        y_train = y_train.detach()

        t = x_train[:, 0:1]
        x0 = x_train[:, 1:5]
        u = x_train[:, 5:7]

        pred = model(t, x0, u)
        mse_loss = loss_fn(pred, y_train)

        res = physics_residual(model, t, x0, u, dt)
        phys_loss = torch.mean(res ** 2)

        total_loss = mse_loss + 1.0 * phys_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: MSE={mse_loss.item():.4f}, Phys={phys_loss.item():.4f}")

    return model

def evaluate_model(model, batch_size=10, T=1.0, dt=0.05):
    model.eval()
    with torch.no_grad():
        x_test, y_test = generate_training_batch(batch_size=batch_size, T=T, dt=dt)
        t = x_test[:, 0:1]
        x0 = x_test[:, 1:5]
        u = x_test[:, 5:7]

        y_pred = model(t, x0, u)
        mse = nn.MSELoss()(y_pred, y_test).item()

        print(f"Evaluation MSE on test batch: {mse:.6f}")
        return mse


if __name__ == "__main__":
    trained_model = train_pinn(epochs=10000, batch_size=64, T=1.0, dt=0.05, lr=1e-3)
    print("Training complete. Beginning evaluation...")
    evaluate_model(trained_model, batch_size=10)

    
