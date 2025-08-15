"""
Modified Bicycle RRT* Path Planning with 2D Visualization and 3D Planning
- 2D visualization for better clarity
- 3D planning with time dimension for STL transducer monitoring
- Bicycle kinematic model with nonlinear dynamics
- Real-time path planning capabilities

author: modified implementation
"""

import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Set, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import environment and STL transducer logic
from environment import OBSTACLE_LIST, RAND_AREA, REGION_PREDICATES, STL_FORMULA
from transducer import parse_formula, build_transducer_from_tree, AtomicTransducer


show_animation = True

# ---------------------------
# STL Formula Manager
# ---------------------------
class STLFormulaManager:
    """Manages multiple STL formulas for path planning"""
    
    def __init__(self):
        self.formulas = {}  # name -> (formula_string, transducer)
        self.task_regions = {}  # name -> list of (x, y, radius) regions
    
    def add_formula(self, name: str, formula_string: str, task_regions: List[Tuple[float, float, float]] = None):
        """Add an STL formula for monitoring"""
        try:
            # Parse and build transducer
            tree = parse_formula(formula_string)
            transducer = build_transducer_from_tree(tree)
            self.formulas[name] = (formula_string, transducer)
            if task_regions:
                self.task_regions[name] = task_regions
            print(f"Added STL formula '{name}': {formula_string}")
            return True
        except Exception as e:
            print(f"Error adding formula '{name}': {e}")
            return False
    
    def evaluate_node(self, node, signal_converter) -> Dict[str, bool]:
        """Evaluate all formulas for a given node"""
        results = {}
        # Convert node trajectory to signal format
        signal = signal_converter(node)
        for name, (formula_str, transducer) in self.formulas.items():
            try:
                # Evaluate at the start of the trajectory
                if hasattr(transducer, 'get_final_verdict'):
                    result = transducer.get_final_verdict(signal)
                else:
                    result = transducer.evaluate(signal, 0)
                results[name] = result
            except Exception as e:
                print(f"Error evaluating formula '{name}': {e}")
                results[name] = False
        return results
    
    def check_compliance(self, results: Dict[str, bool]) -> bool:
        """Check if all formulas are satisfied"""
        return all(results.values())

# ===============================
# Bicycle Model Components
# ===============================
@dataclass
class BicycleState:
    """State representation for bicycle model: [x, y, theta, v]"""
    x: float = 0.0      # x position
    y: float = 0.0      # y position
    theta: float = 0.0  # heading angle
    v: float = 0.0      # velocity
    t: float = 0.0      # time (for STL monitoring)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, t: float = 0.0) -> 'BicycleState':
        return cls(arr[0], arr[1], arr[2], arr[3], t)
    
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

@dataclass
class BicycleControl:
    """Control input for bicycle model: [acceleration, steering_angle]"""
    acceleration: float = 0.0  # longitudinal acceleration
    steering_angle: float = 0.0  # steering angle (front wheel)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.acceleration, self.steering_angle])

class BicycleModel:
    """Bicycle kinematic model with integration"""
    
    def __init__(self, wheelbase: float = 2.7, max_speed: float = 30.0, 
                 max_acceleration: float = 5.0, max_steering_angle: float = np.pi/4):
        self.wheelbase = wheelbase
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_steering_angle = max_steering_angle
        
    def dynamics(self, state: BicycleState, control: BicycleControl) -> np.ndarray:
        """Compute state derivatives for bicycle model"""
        # Clamp control inputs
        a = np.clip(control.acceleration, -self.max_acceleration, self.max_acceleration)
        delta = np.clip(control.steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        # Bicycle model dynamics
        x_dot = state.v * np.cos(state.theta)
        y_dot = state.v * np.sin(state.theta)
        theta_dot = (state.v / self.wheelbase) * np.tan(delta)
        v_dot = a
        
        return np.array([x_dot, y_dot, theta_dot, v_dot])
    
    def integrate(self, state: BicycleState, control: BicycleControl, dt: float) -> BicycleState:
        """Integrate dynamics using RK4"""
        state_vec = state.to_array()
        
        # RK4 integration
        k1 = self.dynamics(state, control)
        k2 = self.dynamics(BicycleState.from_array(state_vec + 0.5 * dt * k1, state.t), control)
        k3 = self.dynamics(BicycleState.from_array(state_vec + 0.5 * dt * k2, state.t), control)
        k4 = self.dynamics(BicycleState.from_array(state_vec + dt * k3, state.t), control)
        
        new_state_vec = state_vec + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Clamp velocity
        new_state_vec[3] = np.clip(new_state_vec[3], 0, self.max_speed)
        
        # Normalize angle
        new_state_vec[2] = self.normalize_angle(new_state_vec[2])
        
        return BicycleState.from_array(new_state_vec, state.t + dt)
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def get_vehicle_corners(self, state: BicycleState, length: float = 4.0, width: float = 2.0) -> np.ndarray:
        """Get the four corners of the vehicle for collision checking"""
        # Vehicle corners in local frame (rear axle at origin)
        corners_local = np.array([
            [-length * 0.3, -width/2],  # rear left
            [-length * 0.3, width/2],   # rear right
            [length * 0.7, width/2],    # front right
            [length * 0.7, -width/2]    # front left
        ])
        
        # Rotation matrix
        cos_theta = np.cos(state.theta)
        sin_theta = np.sin(state.theta)
        R = np.array([[cos_theta, -sin_theta],
                     [sin_theta, cos_theta]])
        
        # Transform to global frame
        corners_global = (R @ corners_local.T).T + np.array([state.x, state.y])
        return corners_global

# ===============================
# Enhanced RRT Node for Bicycle Model
# ===============================
class Node:
    """RRT Node for bicycle model with STL monitoring"""
    
    def __init__(self, state: BicycleState):
        self.state = state
        self.x = state.x  # For compatibility with original code
        self.y = state.y
        self.t = state.t
        self.theta = state.theta
        self.v = state.v
        
        # Path information
        self.path_x = []
        self.path_y = []
        self.path_t = []
        self.path_theta = []
        self.path_v = []
        
        # Tree structure
        self.parent = None
        self.children = []
        self.cost = 0.0
        
        # Control information
        self.control_from_parent = None
        self.integration_time = 0.0

# ===============================
# Enhanced RRT* with Bicycle Dynamics
# ===============================
class BicycleRRT:
    """RRT planning with bicycle dynamics and STL monitoring"""
    def __init__(self, start_state: BicycleState, obstacle_list, rand_area,
                 bicycle_model: BicycleModel = None,
                 expand_dis=3.0, path_resolution=0.5, max_iter=500, 
                 allow_speed=3, cover_threshold=35, goal_sample_rate=5,
                 control_duration=1.0, num_control_samples=11,
                 stl_formula: str = None, region_predicates: dict = None):
        """
        Initialize RRT planner with bicycle dynamics and STL monitoring
        """
        self.start = Node(start_state)
        self.rand_area = rand_area
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.allow_speed = allow_speed
        self.cover_threshold = cover_threshold
        self.select_nd = []
        self.goal_sample_rate = goal_sample_rate
        self.control_duration = control_duration
        self.num_control_samples = num_control_samples

        # Bicycle model
        if bicycle_model is None:
            self.bicycle_model = BicycleModel()
        else:
            self.bicycle_model = bicycle_model

        # STL transducer setup
        if stl_formula is None:
            stl_formula = STL_FORMULA
        if region_predicates is None:
            region_predicates = REGION_PREDICATES
        tree = parse_formula(stl_formula)
        self.stl_transducer = build_transducer_from_tree(tree)
        # Only set region params for atomic predicates, not for operators like F, G, etc.
        self._set_region_params_atomic(self.stl_transducer, region_predicates)

        # Control samples for bicycle model
        self.control_samples = self._generate_control_samples()
        
        # Single figure for animation
        self.fig = None
        self.ax = None
        self.cbar = None

    def _set_region_params_atomic(self, transducer, region_predicates):
        """FIXED: Recursively set region parameters for atomic transducers only."""
        # Only process if this is an AtomicTransducer
        if hasattr(transducer, 'predicate') and isinstance(transducer, AtomicTransducer):
            pred = transducer.predicate
            if pred in region_predicates:
                center = region_predicates[pred][:2]
                radius = region_predicates[pred][2]
                transducer.set_region(center, radius)
                print(f"Set region for atomic predicate '{pred}': center={center}, radius={radius}")
        
        # Recursively process children (for compound formulas)
        if hasattr(transducer, 'children'):
            for child in transducer.children:
                self._set_region_params_atomic(child, region_predicates)
        elif hasattr(transducer, 'left') and hasattr(transducer, 'right'):
            # Binary operators
            self._set_region_params_atomic(transducer.left, region_predicates)
            self._set_region_params_atomic(transducer.right, region_predicates)
        elif hasattr(transducer, 'child'):
            # Unary operators
            self._set_region_params_atomic(transducer.child, region_predicates)
    
    def _generate_control_samples(self) -> List[BicycleControl]:
        """Generate control samples for bicycle model"""
        controls = []
        
        # Acceleration samples
        acc_samples = np.linspace(-self.bicycle_model.max_acceleration/2, 
                                 self.bicycle_model.max_acceleration/2, 
                                 self.num_control_samples)
        
        # Steering angle samples
        steer_samples = np.linspace(-self.bicycle_model.max_steering_angle/2,
                                   self.bicycle_model.max_steering_angle/2,
                                   self.num_control_samples)
        
        # Combine samples (simplified set for performance)
        for acc in acc_samples[::2]:  # Every other sample
            for steer in steer_samples[::2]:
                controls.append(BicycleControl(acc, steer))
        
        return controls
    
    def setup_animation_window(self):
        """Setup single animation window (persisted for all animation updates)"""
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            try:
                self.fig.canvas.manager.set_window_title('Bicycle RRT* Planning Progress')
            except Exception:
                pass
            # Setup colorbar only once
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                      norm=plt.Normalize(vmin=0, vmax=self.rand_area[1][2]))
            sm.set_array([])
            self.cbar = self.fig.colorbar(sm, ax=self.ax)
            self.cbar.set_label('Time (s)')
    
    def planning(self, animation=True):
        """RRT path planning with bicycle dynamics and STL monitoring (single persistent window)"""
        self.node_list = [self.start]
        if animation:
            self.setup_animation_window()

        for i in range(self.max_iter):
            print(f"Iteration {i}/{self.max_iter}")

            # Sample random state (3D sampling: x, y, t)
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node_index(self.node_list, rnd_node)
            if nearest_node is None:
                continue

            # Steer with bicycle dynamics (3D extension)
            new_node = self.steer_bicycle(nearest_node, rnd_node)
            if new_node is None:
                continue

            # STL monitoring: check if the new edge (trajectory) satisfies STL
            signal = [new_node.path_x, new_node.path_y]
            times = new_node.path_t
            self.stl_transducer.reset()
            stl_outputs = []
            stl_satisfied = False
            # times =           [0.0, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
            # new_node.path_x = [5.5,   6, 7,  10,  11,  12,  13,  14,  15,  16,   8,   8,   8,   8,   8,   8,  13]
            # new_node.path_y = [6.5, 6.4, 6,   6,   6,   6,   6,   6,   8,  10,  11,  12,  13,  14,  13,  13,  16]
            for t, pt in zip(times, zip(new_node.path_x, new_node.path_y)):
                # Check if point is in any region (for debugging)
                if (pt[0]-17)**2 + (pt[1]-8)**2 < 3**2:  # Example condition
                    print(f"Point {pt} at time {t} is in region A")
                if (pt[0]-13)**2 + (pt[1]-16)**2 < 3**2:  # Example condition
                    print(f"Point {pt} at time {t} is in region B")
                out = self.stl_transducer.step(pt, t)
                stl_outputs.append(out)
                # If the transducer has a 'satisfied' attribute and it is True, mark as satisfied
                if hasattr(self.stl_transducer, 'satisfied') and self.stl_transducer.satisfied:
                    stl_satisfied = True
            stl_valid = all(o is not False for o in stl_outputs)

            print(f"STL check for edge: {stl_outputs} -> valid: {stl_valid}, satisfied: {stl_satisfied}")

            # Check collision and STL compliance
            if self.check_collision_bicycle(new_node, self.obstacle_list) and stl_valid:
                self.node_list.append(new_node)
                new_node.parent = nearest_node
                nearest_node.children.append(new_node)

                # Calculate cost
                d_n = self.calc_distance_bicycle(new_node, nearest_node)
                new_node.cost = new_node.parent.cost + d_n

                # Add to selectable nodes (leaf nodes that really finish the STL task)
                # Only add if STL task is truly satisfied at this node (not just valid so far)
                if stl_satisfied:
                    self.select_nd.append(new_node)

            # Show animation every 10 iterations (persist single window)
            if animation and i % 10 == 0:
                self.update_animation(rnd_node, i)

        # Final visualization (reuse the same window)
        if animation:
            self.show_final_result()
    
    def update_animation(self, rnd_node, iteration):
        """Update the single animation window"""
        self.ax.clear()
        
        # Draw random node
        if rnd_node is not None:
            self.ax.scatter(rnd_node.x, rnd_node.y, c='red', marker='o', s=50, 
                           label=f'Random (t={rnd_node.t:.1f}s)', zorder=5)
        
        # Draw nodes and edges with time-based coloring
        for node in self.node_list:
            time_color = node.t / self.rand_area[1][2]  # Normalize to [0,1]
            self.ax.scatter(node.x, node.y, c=plt.cm.viridis(time_color), 
                           marker='^', s=20, alpha=0.7, zorder=3)
            
            if node.parent:
                self.ax.plot([node.x, node.parent.x], [node.y, node.parent.y], 
                            'b-', alpha=0.3, linewidth=1, zorder=1)
        
        # Draw obstacles
        for (ox, oy, size) in self.obstacle_list:
            circle = plt.Circle((ox, oy), size, color='blue', alpha=0.7, zorder=2)
            self.ax.add_patch(circle)
        
        # Draw task regions if available
        if hasattr(self, 'region_predicates') and self.region_predicates:
            for pred_name, (cx, cy, radius) in self.region_predicates.items():
                circle = plt.Circle((cx, cy), radius, color='yellow', 
                                  alpha=0.3, linestyle='--', zorder=2)
                self.ax.add_patch(circle)
                self.ax.text(cx, cy, pred_name, ha='center', va='center', zorder=4)
        
        # Draw selectable nodes (STL compliant)
        for node in self.select_nd:
            self.ax.scatter(node.x, node.y, c='green', marker='*', s=100, 
                           alpha=0.8, zorder=4)
        
        # Start point
        self.ax.scatter(self.start.x, self.start.y, c='red', marker='x', s=100, 
                       label='Start', zorder=5)
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title(f'Bicycle RRT* Progress - Iteration {iteration} - {len(self.node_list)} nodes, {len(self.select_nd)} valid')
        self.ax.legend()
        
        # Set consistent axis limits
        margin = 2
        self.ax.set_xlim(self.rand_area[0][0] - margin, self.rand_area[1][0] + margin)
        self.ax.set_ylim(self.rand_area[0][1] - margin, self.rand_area[1][1] + margin)
        
        plt.pause(0.01)
    
    def show_final_result(self):
        """Show final result with path"""
        self.ax.clear()
        
        # Draw all nodes (faded)
        for node in self.node_list:
            time_color = node.t / self.rand_area[1][2]
            self.ax.scatter(node.x, node.y, c=plt.cm.viridis(time_color), 
                           marker='^', s=10, alpha=0.3, zorder=1)
        
        # Draw final path
        if len(self.select_nd) > 0:
            # Find best path
            reach_list = [x for x in self.select_nd 
                         if self.calc_distance_bicycle(x, self.start) < 20]
            
            if reach_list:
                distances = [np.linalg.norm(node.state.position() - self.start.state.position()) + 
                            node.cost for node in reach_list]
                minind = distances.index(min(distances))
                best_node = reach_list[minind]
                
                # Draw path
                path_nodes = []
                cn = best_node
                while cn.parent is not None:
                    path_nodes.append(cn)
                    cn = cn.parent
                path_nodes.append(cn)
                
                # Draw path trajectory
                for i, node in enumerate(path_nodes):
                    if node.parent:
                        self.ax.plot([node.x, node.parent.x], [node.y, node.parent.y], 
                                    'r-', linewidth=3, alpha=0.8, zorder=4)
                    
                    # Draw vehicle at key points
                    if i % max(1, len(path_nodes)//5) == 0:
                        self.draw_vehicle_on_ax(self.ax, node.state, alpha=0.7)
                        self.ax.text(node.x + 1, node.y + 1, 
                                    f"t={self.rand_area[1][2] - node.t:.1f}s",
                                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
                                                        facecolor="white", alpha=0.7), zorder=5)
        
        # Draw environment
        for (ox, oy, size) in self.obstacle_list:
            circle = plt.Circle((ox, oy), size, color='blue', alpha=0.7, zorder=2)
            self.ax.add_patch(circle)
        
        # Draw task regions
        if hasattr(self, 'region_predicates') and self.region_predicates:
            for pred_name, (cx, cy, radius) in self.region_predicates.items():
                circle = plt.Circle((cx, cy), radius, color='yellow', 
                                  alpha=0.3, linestyle='--', linewidth=2, zorder=2)
                self.ax.add_patch(circle)
                self.ax.text(cx, cy, pred_name, ha='center', va='center', 
                            fontweight='bold', zorder=4)
        
        # Start point
        self.ax.scatter(self.start.x, self.start.y, c='red', marker='x', s=150, 
                       label='Start', zorder=5)
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title(f'Final Result - {len(self.select_nd)} STL-compliant paths found')
        self.ax.legend()
        
        # Set consistent axis limits
        margin = 2
        self.ax.set_xlim(self.rand_area[0][0] - margin, self.rand_area[1][0] + margin)
        self.ax.set_ylim(self.rand_area[0][1] - margin, self.rand_area[1][1] + margin)
        
        plt.pause(0.1)
    
    def steer_bicycle(self, from_node: Node, to_node: Node) -> Optional[Node]:
        """Steer using bicycle dynamics (3D extension with time)"""
        best_control = None
        best_new_node = None
        min_distance = float('inf')
        
        # Try different controls
        for control in self.control_samples:
            # Integrate bicycle dynamics
            new_state = self.bicycle_model.integrate(
                from_node.state, control, self.control_duration)
            
            # Check if we're getting closer to target (3D distance with time)
            distance = self.calc_distance_states(new_state, to_node.state)
            
            if distance < min_distance:
                min_distance = distance
                best_control = control
                best_new_node = self._create_trajectory_node(
                    from_node, control, self.control_duration)
        
        return best_new_node
    
    def _create_trajectory_node(self, from_node: Node, control: BicycleControl, 
                              duration: float) -> Node:
        """Create a new node with full trajectory using bicycle dynamics"""
        # Generate trajectory
        trajectory_states = []
        current_state = from_node.state
        dt = self.path_resolution / max(current_state.v, 1.0)  # Adaptive time step
        
        time_elapsed = 0.0
        while time_elapsed < duration:
            trajectory_states.append(current_state)
            remaining_time = min(dt, duration - time_elapsed)
            current_state = self.bicycle_model.integrate(current_state, control, remaining_time)
            time_elapsed += remaining_time
        
        # Final state
        trajectory_states.append(current_state)
        
        # Create node
        new_node = Node(current_state)
        new_node.control_from_parent = control
        new_node.integration_time = duration
        
        # Fill trajectory data
        new_node.path_x = [s.x for s in trajectory_states]
        new_node.path_y = [s.y for s in trajectory_states]
        new_node.path_t = [s.t for s in trajectory_states]
        new_node.path_theta = [s.theta for s in trajectory_states]
        new_node.path_v = [s.v for s in trajectory_states]
        
        return new_node
    
    def calc_distance_states(self, state1: BicycleState, state2: BicycleState) -> float:
        """Calculate distance between two bicycle states (3D with time)"""
        pos_dist = np.linalg.norm(state1.position() - state2.position())
        angle_dist = abs(self.bicycle_model.normalize_angle(state1.theta - state2.theta))
        vel_dist = abs(state1.v - state2.v)
        time_dist = abs(state1.t - state2.t)
        
        return pos_dist + 5.0 * angle_dist + 0.5 * vel_dist + 0.1 * time_dist
    
    def calc_distance_bicycle(self, node1: Node, node2: Node) -> float:
        """Calculate distance between nodes using bicycle state"""
        return self.calc_distance_states(node1.state, node2.state)
    
    def check_collision_bicycle(self, node: Node, obstacle_list) -> bool:
        """Check collision for bicycle model using vehicle corners"""
        if node is None:
            return False
        
        # Check collision along trajectory
        for i in range(len(node.path_x)):
            state = BicycleState(node.path_x[i], node.path_y[i], 
                               node.path_theta[i], node.path_v[i], node.path_t[i])
            
            # Get vehicle corners
            corners = self.bicycle_model.get_vehicle_corners(state)
            
            # Check collision with each obstacle
            for (ox, oy, size) in obstacle_list:
                for corner in corners:
                    dx = ox - corner[0]
                    dy = oy - corner[1]
                    d = dx * dx + dy * dy
                    if d <= size ** 2:
                        return False  # Collision
        
        return True  # Safe
    
    def get_random_node(self) -> Node:
        """Generate random node with bicycle state (3D sampling)"""
        if random.randint(0, 100) > self.goal_sample_rate:
            # Random sampling in 3D space (x, y, t)
            x = random.uniform(self.rand_area[0][0], self.rand_area[1][0])
            y = random.uniform(self.rand_area[0][1], self.rand_area[1][1])
            theta = random.uniform(-np.pi, np.pi)
            v = random.uniform(0, self.bicycle_model.max_speed)
            t = random.uniform(self.rand_area[0][2], self.rand_area[1][2])
            
            state = BicycleState(x, y, theta, v, t)
        else:
            # Sample towards goal (simplified)
            state = BicycleState(
                random.uniform(self.rand_area[1][0] - 5, self.rand_area[1][0]),
                random.uniform(self.rand_area[1][1] - 5, self.rand_area[1][1]),
                0, 10, self.rand_area[1][2]
            )
        
        return Node(state)
    
    def get_nearest_node_index(self, node_list, rnd_node):
        """Find nearest node considering bicycle dynamics and time constraints (3D)"""
        # Nodes that are in the past (can reach the random node)
        past_list = [x for x in node_list if x.t < rnd_node.t]
        if not past_list:
            return None
        
        # Check reachability with bicycle constraints
        reach_list = []
        for node in past_list:
            time_available = rnd_node.t - node.t
            max_distance = self.bicycle_model.max_speed * time_available
            actual_distance = np.linalg.norm(node.state.position() - rnd_node.state.position())
            
            if actual_distance <= max_distance and time_available > 0:
                reach_list.append(node)
        
        if not reach_list:
            return None
        
        # Find closest node in 3D space
        distances = [self.calc_distance_bicycle(node, rnd_node) for node in reach_list]
        min_ind = distances.index(min(distances))
        return reach_list[min_ind]
    
    # ===============================
    # 2D Visualization Methods
    # ===============================
    def draw_vehicle_on_ax(self, ax, state: BicycleState, color='red', alpha=0.7):
        """Draw vehicle shape in 2D on specific axes"""
        corners = self.bicycle_model.get_vehicle_corners(state)
        vehicle_polygon = plt.Polygon(corners, color=color, alpha=alpha)
        ax.add_patch(vehicle_polygon)
        
        # Draw orientation arrow
        arrow_length = 2.0
        ax.arrow(state.x, state.y,
                 arrow_length * np.cos(state.theta),
                 arrow_length * np.sin(state.theta),
                 head_width=0.5, head_length=0.3, fc=color, ec=color, alpha=alpha)
    

    def plot_circle_2d(self, x, y, size, color="b"):
        """Plot 2D circle on current axes"""
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        self.ax.plot(xl, yl, color)


# ===============================
# Main Demo Function
# ===============================
def main_bicycle_rrt_demo():
    """Main demonstration of merged bicycle RRT* with STL monitoring and single window animation"""
    print("Starting Bicycle RRT* with STL Monitoring Demo (Single Window Animation)")
    print("=" * 75)
    
    # Use environment and STL from environment.py
    obstacle_list = OBSTACLE_LIST
    rand_area = RAND_AREA

    # Create bicycle model with realistic parameters
    bicycle_model = BicycleModel(
        wheelbase=2.7,           # meters
        max_speed=15.0,          # m/s
        max_acceleration=3.0,    # m/s²
        max_steering_angle=np.pi/4  # 45 degrees max steering
    )

    # Initial state: position (5.5, 6.5), heading=0, velocity=2 m/s, time=0
    start_state = BicycleState(x=5.5, y=6.5, theta=0.0, v=2.0, t=0.0)

    # Create RRT planner with STL monitoring
    rrt = BicycleRRT(
        start_state=start_state,
        obstacle_list=obstacle_list,
        rand_area=rand_area,
        bicycle_model=bicycle_model,
        expand_dis=3.0,           # meters
        path_resolution=0.3,      # meters
        max_iter=5000,            # iterations
        control_duration=1.0,     # seconds per control
        num_control_samples=7,    # reduced for performance
        stl_formula=STL_FORMULA,
        region_predicates=REGION_PREDICATES
    )
    
    # Store region predicates for visualization
    rrt.region_predicates = REGION_PREDICATES
    
    print(f"Planning with bicycle model:")
    print(f"  Wheelbase: {bicycle_model.wheelbase} m")
    print(f"  Max speed: {bicycle_model.max_speed} m/s")
    print(f"  Max acceleration: {bicycle_model.max_acceleration} m/s²")
    print(f"  Max steering: {bicycle_model.max_steering_angle * 180/np.pi:.1f} degrees")
    print(f"  Start state: x={start_state.x}, y={start_state.y}, θ={start_state.theta:.2f}, v={start_state.v}")
    print(f"  Planning space: 3D (x, y, t) with time range [0, {rand_area[1][2]}]s")
    print(f"  Visualization: Single window with real-time updates")
    
    # Run planning
    start_time = time.time()
    path = rrt.planning(animation=show_animation)
    planning_time = time.time() - start_time
    
    print(f"\nPlanning completed in {planning_time:.2f} seconds")
    print(f"Generated {len(rrt.node_list)} nodes in 3D space")
    print(f"Found {len(rrt.select_nd)} STL-compliant selectable nodes")
    
    # Additional analysis
    if len(rrt.node_list) > 0:
        max_time = max(node.t for node in rrt.node_list)
        avg_speed = np.mean([node.v for node in rrt.node_list])
        print(f"Maximum time reached: {max_time:.2f}s")
        print(f"Average speed: {avg_speed:.2f} m/s")
    
    # Show final results
    if show_animation:
        print("\nAnimation complete! Single window shows:")
        print("  - Real-time RRT tree growth with time-based coloring")
        print("  - STL-compliant nodes marked with green stars")
        print("  - Final optimal path with vehicle dynamics")
        print("  - Task regions and obstacles")
        
        # Keep plot open
        print("\nClose the plot window to continue...")
        plt.show()
    
    return rrt


def run_bicycle_rrt_comparison():
    """Run comparison between different bicycle configurations (no animation)"""
    print("\n" + "=" * 60)
    print("COMPARISON: Different Bicycle Configurations")
    print("=" * 60)
    
    # Same environment for both
    obstacle_list = [(8, 12, 2), (15, 8, 1.5)]
    rand_area = [(0, 0, 0), (20, 20, 15)]
    
    configs = [
        {
            'name': 'Agile Car',
            'wheelbase': 2.5,
            'max_speed': 20.0,
            'max_acceleration': 5.0,
            'max_steering': np.pi/3
        },
        {
            'name': 'Standard Car',
            'wheelbase': 2.7,
            'max_speed': 15.0,
            'max_acceleration': 3.0,
            'max_steering': np.pi/4
        },
        {
            'name': 'Large Vehicle',
            'wheelbase': 4.0,
            'max_speed': 10.0,
            'max_acceleration': 2.0,
            'max_steering': np.pi/6
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting {config['name']}:")
        
        bicycle_model = BicycleModel(
            wheelbase=config['wheelbase'],
            max_speed=config['max_speed'],
            max_acceleration=config['max_acceleration'],
            max_steering_angle=config['max_steering']
        )
        
        start_state = BicycleState(x=2, y=2, theta=0, v=3, t=0)
        
        rrt = BicycleRRT(
            start_state=start_state,
            obstacle_list=obstacle_list,
            rand_area=rand_area,
            bicycle_model=bicycle_model,
            max_iter=300
        )
        
        start_time = time.time()
        rrt.planning(animation=False)  # No animation for comparison
        planning_time = time.time() - start_time
        
        results[config['name']] = {
            'time': planning_time,
            'nodes': len(rrt.node_list),
            'valid_paths': len(rrt.select_nd),
            'success_rate': len(rrt.select_nd) / max(1, len(rrt.node_list)) * 100
        }
        
        print(f"   Time: {planning_time:.2f}s")
        print(f"   Nodes: {len(rrt.node_list)}")
        print(f"   Valid paths: {len(rrt.select_nd)}")
        print(f"   Success rate: {results[config['name']]['success_rate']:.1f}%")
    
    # Summary
    print(f"\nSummary:")
    for name, result in results.items():
        print(f"   {name}: {result['time']:.2f}s, {result['success_rate']:.1f}% success")
    
    return results


def demonstrate_stl_monitoring_2d():
    """Demonstrate STL monitoring capabilities with single window visualization"""
    print("\n" + "=" * 60)
    print("STL MONITORING DEMONSTRATION (Single Window)")
    print("=" * 60)
    
    # Create scenario with time-sensitive tasks
    obstacle_list = [(12, 15, 2)]
    rand_area = [(0, 0, 0), (20, 20, 25)]
    
    bicycle_model = BicycleModel(wheelbase=2.5, max_speed=10.0)
    start_state = BicycleState(x=3, y=3, theta=np.pi/4, v=4.0, t=0.0)
    
    rrt_stl = BicycleRRT(
        start_state=start_state,
        obstacle_list=obstacle_list,
        rand_area=rand_area,
        bicycle_model=bicycle_model,
        max_iter=500
    )
    
    rrt_stl.region_predicates = REGION_PREDICATES
    
    print("STL Formula being monitored:")
    print(f"   {STL_FORMULA}")
    print("Region predicates:")
    for name, (x, y, r) in REGION_PREDICATES.items():
        print(f"   {name}: center=({x}, {y}), radius={r}")
    
    # Run with STL monitoring and animation
    start_time = time.time()
    rrt_stl.planning(animation=show_animation)
    stl_time = time.time() - start_time
    
    print(f"\nSTL Planning Results:")
    print(f"   Planning time: {stl_time:.2f}s")
    print(f"   Total nodes: {len(rrt_stl.node_list)}")
    print(f"   STL-compliant nodes: {len(rrt_stl.select_nd)}")
    if len(rrt_stl.node_list) > 0:
        success_rate = len(rrt_stl.select_nd) / len(rrt_stl.node_list) * 100
        print(f"   STL compliance rate: {success_rate:.1f}%")
    
    # Analyze time distribution of compliant nodes
    if len(rrt_stl.select_nd) > 0:
        compliant_times = [node.t for node in rrt_stl.select_nd]
        print(f"   Compliant node times: {min(compliant_times):.1f}s to {max(compliant_times):.1f}s")
    
    return rrt_stl


if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("BICYCLE RRT* WITH SINGLE WINDOW ANIMATION")
    print("=" * 45)
    print("Key Features:")
    print("- 3D planning space (x, y, time) for STL temporal logic")
    print("- Single window animation with real-time updates")
    print("- Bicycle kinematic model with realistic constraints")
    print("- STL transducer monitoring for temporal requirements")
    print("- Time-based coloring and progress indicators")
    print()
    
    # Main demonstration
    main_rrt = main_bicycle_rrt_demo()
    
    # # Optional: Run additional demonstrations
    # if input("\nRun comparison demo? (y/n): ").lower() == 'y':
    #     comparison_results = run_bicycle_rrt_comparison()
    
    # if input("\nRun STL monitoring demo with animation? (y/n): ").lower() == 'y':
    #     stl_demo = demonstrate_stl_monitoring_2d()
    
    print("\nDemo completed!")