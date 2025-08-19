
# --- Imports for RRT and PINN integration ---
import numpy as np
import matplotlib.pyplot as plt
import torch
from stl_rrt import BicycleRRT, BicycleState, BicycleControl, BicycleModel
from environment import EnvironmentConfig
from test_PINN_PI import DNN_PI, rk4_step
import torch.nn as nn

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

# ===============================
# RRT-informed MPC + PINN control loop
# ===============================
import numpy as np
import torch
from scipy.optimize import minimize

# Wrapper to use the trained PINN for one-step prediction with constant control over dt
class PINNStepModel:
    def __init__(self, pinn_model):
        self.model = pinn_model.eval()

    @torch.no_grad()
    def predict_next_state(self, x, u, dt, substeps: int = 1):
        """
        Predict x(t+dt) from x(t) using the PINN by integrating in 'substeps' with constant u.
        The PINN takes (t, x0, u) -> x(t). We emulate small-time propagation.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(u, np.ndarray):
            u = torch.tensor(u, dtype=torch.float32)
        x_curr = x.clone()
        ds = dt / substeps
        for _ in range(substeps):
            t_in = torch.tensor([[ds]], dtype=torch.float32)
            x0_in = x_curr.unsqueeze(0)
            u_in = u.unsqueeze(0)
            x_next = self.model(t_in, x0_in, u_in).squeeze(0)
            x_curr = x_next
        return x_curr

# Simple reference-tracking MPC that uses the PINN for rollout
class ReferenceTrackingMPC:
    def __init__(self, pinn_stepper: PINNStepModel, horizon=15, dt=0.1,
                 u_bounds=((-1.0, 1.0), (-0.5, 0.5)), w_state=1.0, w_ctrl=1e-3, w_smooth=1e-2):
        self.pinn = pinn_stepper
        self.H = horizon
        self.dt = dt
        self.u_bounds = u_bounds
        self.w_state = w_state
        self.w_ctrl = w_ctrl
        self.w_smooth = w_smooth

    def _rollout(self, x0: np.ndarray, u_seq: np.ndarray):
        x_seq = []
        x = torch.tensor(x0, dtype=torch.float32)
        for i in range(self.H):
            u = torch.tensor(u_seq[i], dtype=torch.float32)
            x = self.pinn.predict_next_state(x, u, self.dt, substeps=1)
            x_seq.append(x)
        return torch.stack(x_seq, dim=0)  # (H, 4)

    def _cost(self, u_flat: np.ndarray, x0: np.ndarray, ref_seq: np.ndarray):
        u_seq = u_flat.reshape(self.H, 2)
        x_preds = self._rollout(x0, u_seq)  # (H, 4)
        ref_t = torch.tensor(ref_seq[:, :4], dtype=torch.float32)
        # state tracking (x,y,theta,v)
        state_err = x_preds - ref_t
        J_state = torch.sum(state_err ** 2)
        # control effort and smoothness
        u_t = torch.tensor(u_seq, dtype=torch.float32)
        J_ctrl = torch.sum(u_t ** 2)
        J_smooth = torch.sum((u_t[1:] - u_t[:-1]) ** 2) if self.H > 1 else torch.tensor(0.0)
        J = self.w_state * J_state + self.w_ctrl * J_ctrl + self.w_smooth * J_smooth
        return float(J.item())

    def optimize(self, x0: np.ndarray, ref_seq: np.ndarray):
        u0 = np.zeros((self.H, 2), dtype=np.float64)
        bounds = list(self.u_bounds) * self.H
        res = minimize(self._cost, u0.flatten(), args=(x0, ref_seq), method='L-BFGS-B', bounds=bounds)
        if not res.success:
            # fallback: small acceleration towards heading, zero steer
            return u0
        return res.x.reshape(self.H, 2)

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
    Sample (at most) 'horizon' waypoints ahead starting from start_idx. If the path has timestamps t,
    we align by index; otherwise we just take subsequent points. Returns an array (H, 5): [x,y,theta,v,t].
    If not enough points remain, pad with the last one.
    """
    end_idx = min(start_idx + horizon, len(path))
    ref = path[start_idx:end_idx]
    if ref.shape[0] < horizon:
        last = ref[-1]
        pad = np.repeat(last[None, :], horizon - ref.shape[0], axis=0)
        ref = np.vstack([ref, pad])
    return ref

# --------- Closed-loop control using RRT reference + PINN-MPC ---------

def rrt_pinn_mpc_control_loop(rrt_paths, trained_pinn_model, initial_state, steps=80, horizon=15, dt=0.1,
                              use_true_simulator=True):
    """
    rrt_paths: list of numpy arrays, each of shape (N_i, 5) with columns [x,y,theta,v,t]
    trained_pinn_model: the PINN instance previously trained
    initial_state: np.array(4,) = [x,y,theta,v]
    use_true_simulator: if True, propagate the real system with rk4_step; else use PINN for propagation
    Returns history dict with states, controls, references.
    """
    pinn_stepper = PINNStepModel(trained_pinn_model)
    mpc = ReferenceTrackingMPC(pinn_stepper, horizon=horizon, dt=dt)

    x = np.array(initial_state, dtype=np.float32)
    states = [x.copy()]
    controls = []
    refs_logged = []

    # choose the best path for the changed initial state
    best_path = find_best_rrt_path(rrt_paths, x[:2])
    if best_path is None:
        raise ValueError("No RRT paths provided.")
    best_path = np.asarray(best_path, dtype=np.float32)

    for k in range(steps):
        # find nearest point along the chosen path to current position
        idx = nearest_index_on_path(best_path, x[:2])
        # get reference window for the MPC horizon
        ref_window = sample_reference_from_path(best_path, idx, horizon, dt)  # (H, 5)
        refs_logged.append(ref_window)

        # optimize control sequence to track ref
        u_seq = mpc.optimize(x, ref_window)
        u0 = u_seq[0]
        controls.append(u0.copy())

        # propagate the system (choose ground-truth or PINN)
        if use_true_simulator:
            x_t = torch.tensor(x, dtype=torch.float32)
            u_t = torch.tensor(u0, dtype=torch.float32)
            x_next = rk4_step(x_t, u_t, dt).detach().numpy()
        else:
            x_next = pinn_stepper.predict_next_state(torch.tensor(x, dtype=torch.float32),
                                                     torch.tensor(u0, dtype=torch.float32), dt).detach().numpy()
        x = x_next
        states.append(x.copy())

        # (optional) break if near the end of path
        if idx >= len(best_path) - 2:
            break

    return {
        "states": np.array(states),
        "controls": np.array(controls),
        "references": np.array(refs_logged, dtype=object)
    }


if __name__ == "__main__":
    # 1. Generate STL-compliant RRT path
    print("Generating STL-compliant RRT path...")
    rrt_path = generate_rrt_stl_path()  # shape (N, 5)
    print(f"RRT path shape: {rrt_path.shape}")

    # 2. Load trained PINN model using PINN_PI_Module from test_PINN_PI.py
    print("Loading trained PINN model...")
    from test_PINN_PI import PINN_PI_Module
    checkpoint_path = 'trained_model.pt'
    try:
        pinn_module = PINN_PI_Module(seed=0, gpu=0, checkpoint_path=checkpoint_path)
        pinn_model = pinn_module.get_model()
        print("Loaded trained PINN model from", checkpoint_path)
    except Exception as e:
        print("Warning: Could not load trained_model.pt, using untrained PINN. Error:", e)
        pinn_module = PINN_PI_Module(seed=0, gpu=0)
        pinn_model = pinn_module.get_model()

    # 3. Run closed-loop control using RRT reference and PINN-MPC
    print("Running closed-loop control with RRT reference and PINN-MPC...")
    initial_state = rrt_path[0, :4]  # [x, y, theta, v]
    rrt_paths = [rrt_path]
    out = rrt_pinn_mpc_control_loop(
        rrt_paths=rrt_paths,
        trained_pinn_model=pinn_model,
        initial_state=initial_state,
        steps=80,
        horizon=15,
        dt=0.1,
        use_true_simulator=True
    )

    # 4. Plot results
    print("Plotting closed-loop trajectory...")
    states = out['states']
    plt.figure(figsize=(10, 6))
    plt.plot(rrt_path[:, 0], rrt_path[:, 1], 'b--', label='RRT STL Path')
    plt.plot(states[:, 0], states[:, 1], 'r-', label='Closed-loop Trajectory')
    plt.scatter(rrt_path[0, 0], rrt_path[0, 1], c='g', marker='o', label='Start')
    plt.scatter(rrt_path[-1, 0], rrt_path[-1, 1], c='k', marker='*', label='Goal/Leaf')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Closed-loop Trajectory Tracking with RRT+STL+PINN-MPC')
    plt.legend()
    plt.grid()
    plt.show()
