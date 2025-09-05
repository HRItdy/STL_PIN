# --- Imports for RRT and PINN integration ---
import numpy as np
import matplotlib.pyplot as plt
import torch
from stl_rrt import BicycleRRT, BicycleState, BicycleControl, BicycleModel
from PINN_PI import PINN_PI, generate_training_batch
import torch.nn as nn
from scipy.interpolate import interp1d
import torch.backends.cudnn as cudnn
import time

# --------------------------------------------
# RRT + STL + PINN-MPC integration pipeline
# --------------------------------------------
def generate_rrt_stl_path():
    """
    Generate an STL-compliant RRT path using the canonical environment.py globals.
    """
    from environment import OBSTACLE_LIST, RAND_AREA, REGION_PREDICATES, STL_FORMULA
    from stl_rrt import BicycleRRT, BicycleState, BicycleModel

    # Initial state: [x, y, theta, v, t]
    start_state = BicycleState(x=5.5, y=6.5, theta=0.0, v=2.0, t=0.0)

    # Create bicycle model
    bicycle_model = BicycleModel(
        wheelbase=2.7,
        max_speed=15.0,
        max_acceleration=3.0,
        max_steering_angle=np.pi/4
    )

    # Create RRT planner with STL monitoring
    rrt = BicycleRRT(
        start_state=start_state,
        obstacle_list=OBSTACLE_LIST,
        rand_area=RAND_AREA,
        bicycle_model=bicycle_model,
        expand_dis=3.0,
        path_resolution=0.3,
        max_iter=1000,
        control_duration=1.0,
        num_control_samples=7,
        stl_formula=STL_FORMULA,
        region_predicates=REGION_PREDICATES
    )
    rrt.region_predicates = REGION_PREDICATES
    rrt.planning(animation=False)

    # Extract STL-compliant leaf node path (if any)
    if not rrt.select_nd:
        raise RuntimeError("No STL-compliant path found by RRT.")
    # Pick the best (lowest cost) STL-compliant node
    best_node = min(rrt.select_nd, key=lambda n: n.cost)
    # Reconstruct path from root to this node
    path_nodes = []
    cn = best_node
    while cn.parent is not None:
        path_nodes.append(cn)
        cn = cn.parent
    path_nodes.append(cn)
    path_nodes.reverse()
    # Convert to numpy array (N, 5): [x, y, theta, v, t]
    path_arr = np.array([[n.x, n.y, n.theta, n.v, n.t] for n in path_nodes], dtype=np.float32)
    return path_arr

def interpolate_rrt_path(path, target_dt=0.1):
    """
    Interpolate sparse RRT path to create dense reference trajectory.
    
    Args:
        path: numpy array of shape (N, 5) with columns [x, y, theta, v, t]
        target_dt: desired time step for interpolated path
    
    Returns:
        interpolated_path: numpy array of shape (M, 5) with denser waypoints
    """
    if len(path) < 2:
        return path
    
    # Extract time values
    t_original = path[:, 4]  # time column
    t_start, t_end = t_original[0], t_original[-1]
    
    # Create new time grid with target_dt resolution
    t_new = np.arange(t_start, t_end + target_dt, target_dt)
    
    # Ensure we don't go beyond the original end time
    t_new = t_new[t_new <= t_end]
    if t_new[-1] < t_end:
        t_new = np.append(t_new, t_end)
    
    # Handle case where original path has duplicate time values
    if len(np.unique(t_original)) < len(t_original):
        # Add small offsets to make times unique
        for i in range(1, len(t_original)):
            if t_original[i] <= t_original[i-1]:
                t_original[i] = t_original[i-1] + 1e-6
    
    interpolated_path = np.zeros((len(t_new), 5))
    interpolated_path[:, 4] = t_new  # time column
    
    # Interpolate each state variable
    for i in range(4):  # x, y, theta, v
        if i == 2:  # theta (heading) - handle angle wrapping
            interpolated_path[:, i] = interpolate_angle(t_original, path[:, i], t_new)
        else:
            # Linear interpolation for x, y, v
            if len(np.unique(t_original)) > 1:
                interp_func = interp1d(t_original, path[:, i], kind='linear', 
                                     bounds_error=False, fill_value='extrapolate')
                interpolated_path[:, i] = interp_func(t_new)
            else:
                # Constant value if only one unique time point
                interpolated_path[:, i] = path[0, i]
    
    return interpolated_path.astype(np.float32)

def interpolate_angle(t_old, theta_old, t_new):
    """
    Interpolate angles handling wraparound at -pi/pi.
    """
    if len(np.unique(t_old)) <= 1:
        return np.full_like(t_new, theta_old[0])
    
    # Unwrap angles to handle discontinuities
    theta_unwrapped = np.unwrap(theta_old)
    
    # Interpolate unwrapped angles
    interp_func = interp1d(t_old, theta_unwrapped, kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
    theta_new_unwrapped = interp_func(t_new)
    
    # Wrap back to [-pi, pi]
    theta_new = np.arctan2(np.sin(theta_new_unwrapped), np.cos(theta_new_unwrapped))
    
    return theta_new

def smooth_velocity_profile(path, max_accel=2.0, dt=0.1):
    """
    Smooth velocity profile to respect acceleration constraints.
    
    Args:
        path: numpy array of shape (N, 5) with columns [x, y, theta, v, t]
        max_accel: maximum allowed acceleration magnitude
        dt: time step
    
    Returns:
        path with smoothed velocity profile
    """
    if len(path) < 2:
        return path
    
    path_smooth = path.copy()
    
    for i in range(1, len(path)):
        dt_actual = path[i, 4] - path[i-1, 4]
        if dt_actual <= 0:
            continue
            
        v_prev = path_smooth[i-1, 3]
        v_desired = path[i, 3]
        
        # Calculate maximum velocity change allowed
        max_dv = max_accel * dt_actual
        
        # Limit velocity change
        dv = v_desired - v_prev
        if abs(dv) > max_dv:
            dv = np.sign(dv) * max_dv
        
        path_smooth[i, 3] = v_prev + dv
        
        # Ensure non-negative velocity
        path_smooth[i, 3] = max(0.0, path_smooth[i, 3])
    
    return path_smooth

def generate_interpolated_rrt_path(target_dt=0.1, smooth_velocity=True):
    """
    Generate and interpolate RRT path for MPC control.
    
    Args:
        target_dt: desired time resolution for interpolated path
        smooth_velocity: whether to apply velocity smoothing
    
    Returns:
        interpolated and optionally smoothed path
    """
    # Generate original sparse RRT path
    print("Generating sparse RRT path...")
    sparse_path = generate_rrt_stl_path()
    print(f"Original RRT path has {len(sparse_path)} waypoints over {sparse_path[-1,4]:.2f} seconds")
    
    # Interpolate to create dense reference
    print(f"Interpolating path with dt={target_dt}...")
    dense_path = interpolate_rrt_path(sparse_path, target_dt=target_dt)
    print(f"Interpolated path has {len(dense_path)} waypoints")
    
    # Optionally smooth velocity profile
    if smooth_velocity:
        print("Smoothing velocity profile...")
        dense_path = smooth_velocity_profile(dense_path, max_accel=2.0, dt=target_dt)
    
    return sparse_path, dense_path

# ===============================
# RRT-informed MPC + PINN control loop
# ===============================
from scipy.optimize import minimize

# Simple reference-tracking MPC that uses the PINN for rollout
class ReferenceTrackingMPC:
    def __init__(self, pinn_stepper, horizon=15, dt=0.1,
                 u_bounds=((-1.0, 1.0), (-0.5, 0.5)), w_state=1.0, w_ctrl=1e-3, w_smooth=1e-2):
        self.pinn = pinn_stepper
        self.H = horizon
        self.dt = dt
        self.u_bounds = u_bounds
        self.w_state = w_state
        self.w_ctrl = w_ctrl
        self.w_smooth = w_smooth
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")

    def _rollout(self, x0: np.ndarray, u_seq: np.ndarray):
        x_seq = []
        x = np.array(x0, dtype=np.float32)
        for i in range(self.H):
            u = np.array(u_seq[i], dtype=np.float32)
            batch_size = 1
            T = 5.0
            dt = 0.05
            n_nodes_list = [256, 256, 4]
            # Generate training batch as numpy arrays
            x, y = generate_training_batch(x, u, batch_size, T, dt, False)
            x_all, y_all = x.cpu().numpy(), y.cpu().numpy()
            _ic = x_all[:, 0] == 0
            x_ic = x_all[_ic]
            y_ic = y_all[_ic]
            pred = self.pinn.infer(x_all, y_all, x_ic, y_ic, n_nodes_list, self.device)
            pred_np = np.array(pred).squeeze()
            x = pred_np[0] if len(pred_np.shape) > 1 else pred_np
            x_seq.append(x.copy())
        return np.stack(x_seq)

    def _cost(self, u_flat: np.ndarray, x0: np.ndarray, ref_seq: np.ndarray):
        u_seq = u_flat.reshape(self.H, 2).astype(np.float32)
        x0_np = np.array(x0, dtype=np.float32)
        ref_np = np.array(ref_seq[:, :4], dtype=np.float32)
        x_preds = self._rollout(x0_np, u_seq)
        state_err = x_preds - ref_np
        J_state = np.sum(state_err ** 2)
        J_ctrl = np.sum(u_seq ** 2)
        if self.H > 1:
            J_smooth = np.sum((u_seq[1:] - u_seq[:-1]) ** 2)
        else:
            J_smooth = 0.0
        J = self.w_state * J_state + self.w_ctrl * J_ctrl + self.w_smooth * J_smooth
        return float(J)

    def optimize(self, x0: np.ndarray, ref_seq: np.ndarray):
        u0 = np.zeros((self.H, 2), dtype=np.float64)
        bounds = list(self.u_bounds) * self.H
        res = minimize(
                self._cost, 
                u0.flatten(), 
                args=(x0, ref_seq),
                method='SLSQP', 
                bounds=bounds,
                options={'maxiter': 10, 'disp': False}
        )
        if res.success:
            return res.x.reshape(self.H, 2).astype(np.float32)
        else:
            print("MPC optimization failed, using zero controls.")
            return np.zeros((self.H, 2), dtype=np.float32)

# -------- Utilities to use an RRT trajectory as reference --------

def find_best_rrt_path(rrt_paths, current_state_xy):
    """Pick the path whose first waypoint is closest (by xy) to the current state."""
    best, best_d = None, float('inf')
    cx, cy = current_state_xy
    for path in rrt_paths:
        if len(path) == 0:
            continue
        d = np.hypot(path[0][0] - cx, path[0][1] - cy)
        if d < best_d:
            best_d, best = d, path
    return best

def nearest_index_on_path(path: np.ndarray, state_xy: np.ndarray):
    diffs = path[:, :2] - state_xy[None, :]
    d2 = np.sum(diffs**2, axis=1)
    return int(np.argmin(d2))

def sample_reference_from_path(path: np.ndarray, start_idx: int, horizon: int, dt: float):
    """
    Sample (at most) 'horizon' waypoints ahead starting from start_idx. 
    For interpolated paths, we can use time-based sampling for better alignment.
    """
    if start_idx >= len(path):
        # If we're beyond the path, return the last waypoint repeated
        last_waypoint = path[-1:].repeat(horizon, axis=0)
        return last_waypoint
    
    # Time-based sampling: look ahead by horizon * dt seconds
    current_time = path[start_idx, 4]  # time at current reference point
    target_end_time = current_time + horizon * dt
    
    # Find all waypoints within the time window
    time_mask = (path[:, 4] >= current_time) & (path[:, 4] <= target_end_time)
    ref_candidates = path[time_mask]
    
    if len(ref_candidates) == 0:
        # Fallback to index-based sampling
        end_idx = min(start_idx + horizon, len(path))
        ref = path[start_idx:end_idx]
    else:
        # Use time-based candidates and subsample to match horizon
        if len(ref_candidates) > horizon:
            # Subsample evenly
            indices = np.linspace(0, len(ref_candidates)-1, horizon, dtype=int)
            ref = ref_candidates[indices]
        else:
            ref = ref_candidates
    
    # Pad with last waypoint if needed
    if len(ref) < horizon:
        last = ref[-1] if len(ref) > 0 else path[start_idx]
        pad = np.repeat(last[None, :], horizon - len(ref), axis=0)
        ref = np.vstack([ref, pad]) if len(ref) > 0 else pad
    
    return ref


# --------- Closed-loop control using RRT reference + PINN-MPC ---------

def find_nearest_time_index(path: np.ndarray, current_time: float):
    """Find the index in the path whose time stamp is closest to current_time."""
    time_diffs = np.abs(path[:, 4] - current_time)
    return int(np.argmin(time_diffs))

def rrt_pinn_mpc_control_loop(rrt_paths, initial_state, steps=80, horizon=15, dt=0.1):
    """
    rrt_paths: list of numpy arrays, each of shape (N_i, 5) with columns [x,y,theta,v,t]
    initial_state: np.array(4,) = [x,y,theta,v]
    Returns history dict with states, controls, references.
    """
    
    # Initialize the trained PINN
    gpu = 0
    seed = 0
    # pred_all = PINN_PI(seed=0, gpu=0)
    # print(f"Prediction shape: {np.array(pred_all).shape}")
    
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

    DNN = PINN_PI(seed=0, gpu=0)
        
    mpc = ReferenceTrackingMPC(pinn_stepper=DNN, horizon=horizon, dt=dt)

    x = torch.tensor(initial_state, dtype=torch.float32, device=device)
    states = [x.clone()]
    controls = []
    refs_logged = []

    best_path = find_best_rrt_path(rrt_paths, x[:2].cpu().numpy())
    if best_path is None:
        raise ValueError("No RRT paths provided.")
    best_path = torch.tensor(best_path, dtype=torch.float32, device=device)

    mpc_times = []
    deviations = []

    # Start at the first time stamp
    current_time = best_path[0, 4].item()

    for k in range(steps):
        print(f"Step {k+1}/{steps}, current state: {x.cpu().numpy()}")

        # Find the reference index by time, not by spatial proximity
        idx = find_nearest_time_index(best_path.cpu().numpy(), current_time)

        # Sample reference window starting from idx
        ref_window = sample_reference_from_path(best_path.cpu().numpy(), idx, horizon, dt)
        ref_window = torch.tensor(ref_window, dtype=torch.float32, device=device)
        refs_logged.append(ref_window.clone())

        # Time the MPC optimization
        t_mpc_start = time.time()
        u_seq = mpc.optimize(x.cpu().numpy(), ref_window.cpu().numpy())
        t_mpc_end = time.time()
        mpc_times.append(t_mpc_end - t_mpc_start)

        u0 = u_seq[0]
        controls.append(torch.tensor(u0, dtype=torch.float32, device=device))

        # Forward simulation
        batch_size = 1
        T = 5.0
        dt_sim = dt
        x, y = generate_training_batch(x, u0, batch_size, T, dt_sim, False)
        x_all, y_all = x.to(device), y.to(device)
        _ic = x_all[:, 0] == 0
        x_ic = x_all[_ic]
        y_ic = y_all[_ic]
        pred = DNN.infer(x_all, y_all, x_ic, y_ic, [256, 256, 4], device)
        pred_tensor = torch.tensor(pred, dtype=torch.float32, device=device).squeeze()
        x = pred_tensor[0]
        states.append(x.clone())

        # Advance the time by dt for next step
        current_time += dt

        # Calculate deviation from reference path at current step
        ref_xy = best_path[idx, :2].cpu().numpy()
        x_xy = x[:2].cpu().numpy()
        deviation = np.linalg.norm(x_xy - ref_xy)
        deviations.append(deviation)

        if idx >= len(best_path) - 2:
            break

    return {
        "states": torch.stack(states).cpu().numpy(),
        "controls": torch.stack(controls).cpu().numpy(),
        "references": [r.cpu().numpy() for r in refs_logged],
        "mpc_times": np.array(mpc_times),
        "deviations": np.array(deviations)
    }
    
# def test_pinn_accuracy(pinn_model, x0, u, dt=0.1, steps=20, device='cpu'):
    """
    Compare PINN_PI predictions to ground truth (RK4 bicycle model) for a constant control input.
    Prints and returns the MSE loss over the trajectory.
    """
    # Import the true bicycle model from PINN_PI.py
    from PINN_PI import rk4_step

    # Prepare initial state and control
    batch_size = 1
    T = 5.0
    dt = 0.05
    x_train, y_train = generate_training_batch(batch_size, T, dt)
    x_train_all, y_train_all = x_train.detach().cpu().numpy(), y_train.detach().cpu().numpy()
    _ic = x_train_all[:, 0] == 0
    x_train_ic = x_train_all[_ic]
    y_train_ic = y_train_all[_ic]
    x_true = torch.tensor(x0, dtype=torch.float32, device=device)
    x_pinn = torch.tensor(x0, dtype=torch.float32, device=device)
    u_tensor = torch.tensor(u, dtype=torch.float32, device=device)

    true_traj = [x_true.cpu().numpy()]
    pinn_traj = [x_pinn.cpu().numpy()]

    for _ in range(steps):
        # Ground truth with RK4
        x_true = rk4_step(x_true, u_tensor, dt)
        true_traj.append(x_true.cpu().numpy())

        # PINN prediction
        x_pinn = pinn_model.predict_next_state(x_pinn, u_tensor, dt)
        pinn_traj.append(x_pinn.cpu().numpy())

    true_traj = np.stack(true_traj)
    pinn_traj = np.stack(pinn_traj)

    mse = np.mean((true_traj - pinn_traj) ** 2)
    print(f"PINN vs Ground Truth MSE over {steps} steps: {mse:.6f}")

    # Optionally plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(true_traj[:, 0], true_traj[:, 1], 'k-', label='Ground Truth')
    plt.plot(pinn_traj[:, 0], pinn_traj[:, 1], 'r--', label='PINN Prediction')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('PINN vs Ground Truth Trajectory')
    plt.show()

    return mse


if __name__ == "__main__":
    # Test PINN accuracy
    # checkpoint_path= 'trained_model.pt'
    # pinn_model = PINN_PI(checkpoint_path=checkpoint_path)
    # x0 = [0.0, 0.0, 0.0, 1.0]  # initial state: [x, y, theta, v]
    # u = [1.0, 0.1]             # constant control: [a, delta]
    # test_pinn_accuracy(pinn_model, x0, u, dt=0.05, steps=20)
    # print("PINN accuracy test completed.")
    # 1. Generate and interpolate STL-compliant RRT path
    print("Generating and interpolating STL-compliant RRT path...")
    sparse_path, dense_path = generate_interpolated_rrt_path(target_dt=0.1, smooth_velocity=True)
    print(f"Dense path shape: {dense_path.shape}")

    # 2. Run closed-loop control using interpolated RRT reference and PINN-MPC
    print("Running closed-loop control with interpolated RRT reference and PINN-MPC...")
    initial_state = dense_path[0, :4]  # [x, y, theta, v]
    rrt_paths = [dense_path]  # Use interpolated path
    out = rrt_pinn_mpc_control_loop(
        rrt_paths=rrt_paths,
        initial_state=initial_state,
        steps=20,
        horizon=1,
        dt=0.05
    )

    # 4. Plot results comparing sparse vs dense paths
    print("Plotting results...")
    states = out['states']
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Trajectory comparison
    plt.subplot(2, 2, 1)
    plt.plot(sparse_path[:, 0], sparse_path[:, 1], 'b--', linewidth=2, label='Original Sparse RRT Path')
    plt.plot(dense_path[:, 0], dense_path[:, 1], 'g:', linewidth=1.5, label='Interpolated Dense Path')
    plt.plot(states[:, 0], states[:, 1], 'r-', linewidth=2, label='Closed-loop Trajectory')
    plt.scatter(sparse_path[0, 0], sparse_path[0, 1], c='g', marker='o', s=100, label='Start')
    plt.scatter(sparse_path[-1, 0], sparse_path[-1, 1], c='k', marker='*', s=200, label='Goal')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Trajectory Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Plot 2: Velocity profiles
    plt.subplot(2, 2, 2)
    plt.plot(sparse_path[:, 4], sparse_path[:, 3], 'b--o', label='Sparse RRT Velocity')
    plt.plot(dense_path[:, 4], dense_path[:, 3], 'g:', label='Interpolated Velocity')
    plt.plot(np.arange(len(states)) * 0.1, states[:, 3], 'r-', label='Actual Velocity')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Velocity Profile Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Path density comparison
    plt.subplot(2, 2, 3)
    sparse_distances = np.diff(sparse_path[:, :2], axis=0)
    sparse_step_sizes = np.linalg.norm(sparse_distances, axis=1)
    dense_distances = np.diff(dense_path[:, :2], axis=0)
    dense_step_sizes = np.linalg.norm(dense_distances, axis=1)
    
    plt.hist(sparse_step_sizes, bins=20, alpha=0.7, label=f'Sparse (avg: {np.mean(sparse_step_sizes):.3f}m)', color='blue')
    plt.hist(dense_step_sizes, bins=20, alpha=0.7, label=f'Dense (avg: {np.mean(dense_step_sizes):.3f}m)', color='green')
    plt.xlabel('Step Size [m]')
    plt.ylabel('Frequency')
    plt.title('Path Step Size Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Control inputs
    plt.subplot(2, 2, 4)
    controls = out['controls']
    time_controls = np.arange(len(controls)) * 0.1
    plt.plot(time_controls, controls[:, 0], 'r-', label='Acceleration')
    plt.plot(time_controls, controls[:, 1], 'b-', label='Steering')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Input')
    plt.title('Control Inputs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 5. Plot actual closed-loop trajectory in a separate figure
    plt.figure(figsize=(8, 6))
    plt.plot(states[:, 0], states[:, 1], 'r-', linewidth=2, label='Closed-loop Trajectory')
    plt.scatter(states[0, 0], states[0, 1], c='g', marker='o', s=100, label='Start')
    plt.scatter(states[-1, 0], states[-1, 1], c='k', marker='*', s=200, label='End')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Actual Closed-loop Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()
    
    mpc_times = out['mpc_times']
    deviations = out['deviations']
    print(f"Control loop completed successfully!")
    print(f"Original sparse path: {len(sparse_path)} waypoints")
    print(f"Interpolated dense path: {len(dense_path)} waypoints")
    print(f"Final position error: {np.linalg.norm(states[-1][:2] - dense_path[-1][:2]):.3f} m")
    print(f"Average MPC optimization time per step: {np.mean(mpc_times):.4f} seconds")
    print(f"Max deviation from reference path: {np.max(deviations):.4f} m")
    print(f"Mean deviation from reference path: {np.mean(deviations):.4f} m")