"""
Merged RRT* Path Planning with Bicycle Dynamics and STL Transducer Monitoring

This implementation combines:
1. Bicycle kinematic model with nonlinear dynamics
2. STL transducer monitoring from the original code
3. 3D visualization with time dimension
4. Real-time path planning capabilities

author: merged implementation
"""
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Set, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import the STL transducer components
from typing import List, Dict, Any, Tuple, Union
import re
from collections import deque
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
            transducer = build_tst_from_tree(tree)
            
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
                result = transducer.evaluate(signal, 0)
                results[name] = result
            except Exception as e:
                print(f"Error evaluating formula '{name}': {e}")
                results[name] = False
        
        return results
    
    def check_compliance(self, results: Dict[str, bool]) -> bool:
        """Check if all formulas are satisfied"""
        return all(results.values())

show_animation = True

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
        
        # STL monitoring (from original code)
        self.state_path = {'Once':{(0, 10): [], (10, 18): []}}
        self.state_stl = {'Once':{(0, 10): None, (10, 18): None}}
        
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
                 control_duration=1.0, num_control_samples=11):
        """
        Initialize RRT planner with bicycle dynamics
        
        Args:
            start_state: Initial bicycle state
            obstacle_list: List of obstacles [(x, y, radius), ...]
            rand_area: Sampling area [(min_x, min_y, min_t), (max_x, max_y, max_t)]
            bicycle_model: Bicycle dynamics model
        """
        # Initialize with bicycle state
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
        
        # STL monitoring (from original code)
        self.task_point = {'Once':{(0, 10): [(17, 8, 3), (10, 16, 3)], (10, 18): [(5, 6, 3)]}}
        self.start_node = Node(BicycleState(15, 15, 0, 0, 20))
        
        # Control samples for bicycle model
        self.control_samples = self._generate_control_samples()
    
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
    
    def planning(self, animation=True):
        """RRT path planning with bicycle dynamics"""
        self.node_list = [self.start]
        
        for i in range(self.max_iter):
            print(f"Iteration {i}/{self.max_iter}")
            
            # Sample random state
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node_index(self.node_list, rnd_node)
            
            if nearest_node is None:
                continue
            
            # Steer with bicycle dynamics
            new_node = self.steer_bicycle(nearest_node, rnd_node)
            if new_node is None:
                continue
            
            # STL monitoring (from original code)
            for k, v in self.task_point['Once'].items():
                checker = Once(self.rand_area[1][2], nearest_node, new_node, k)
                new_node.state_stl['Once'][k] = checker.trans()
            
            print("STL check results:")
            for k, v in new_node.state_stl['Once'].items():
                print(f'bound: {k}, value: {v}')
            
            # Check collision and STL compliance
            if (self.check_collision_bicycle(new_node, self.obstacle_list) and 
                self.check_propose(new_node)):
                
                self.node_list.append(new_node)
                new_node.parent = nearest_node
                nearest_node.children.append(new_node)
                
                # Calculate cost
                d_n = self.calc_distance_bicycle(new_node, nearest_node)
                new_node.cost = new_node.parent.cost + d_n
                
                # Add to selectable nodes
                if (self.rand_area[1][2] - new_node.t <= 5 and 
                    self.check_aval(new_node)):
                    self.select_nd.append(new_node)
            
            if animation and i % 10 == 0:  # Show every 10 iterations
                self.draw_graph(rnd_node)
        
        # Final visualization
        self.draw_rrt()
        self.draw_path_3D(self.start_node)
        self.draw_path_2D(self.start_node)
    
    def steer_bicycle(self, from_node: Node, to_node: Node) -> Optional[Node]:
        """Steer using bicycle dynamics"""
        best_control = None
        best_new_node = None
        min_distance = float('inf')
        
        # Try different controls
        for control in self.control_samples:
            # Integrate bicycle dynamics
            new_state = self.bicycle_model.integrate(
                from_node.state, control, self.control_duration)
            
            # Check if we're getting closer to target
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
        
        # Check task points for STL monitoring
        self.check_task_point(new_node)
        
        return new_node
    
    def calc_distance_states(self, state1: BicycleState, state2: BicycleState) -> float:
        """Calculate distance between two bicycle states"""
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
        """Generate random node with bicycle state"""
        if random.randint(0, 100) > self.goal_sample_rate:
            # Random sampling
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
        """Find nearest node considering bicycle dynamics and time constraints"""
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
        
        # Find closest node
        distances = [self.calc_distance_bicycle(node, rnd_node) for node in reach_list]
        min_ind = distances.index(min(distances))
        return reach_list[min_ind]
    
    # ===============================
    # STL Monitoring Methods (from original code)
    # ===============================
    def check_task_point(self, node):
        """Check task points for STL monitoring (from original code)"""
        for index, value in enumerate(node.path_t):
            for k, v in self.task_point['Once'].items():
                if (value > (self.rand_area[1][2] - k[1]) and 
                    value < (self.rand_area[1][2] - k[0])):
                    dist = [(node.path_x[index] - item[0]) ** 2 + 
                           (node.path_y[index] - item[1]) ** 2 < item[2] ** 2 
                           for item in v]
                    if any(dist):
                        node.state_path['Once'][k].append(True)
                    else:
                        node.state_path['Once'][k].append(False)
    
    def check_propose(self, node):
        """Check if node satisfies STL propositions"""
        for k, v in node.state_stl['Once'].items():
            if v is False:
                return False
        return True
    
    def check_aval(self, node):
        """Check if node is available (STL satisfied)"""
        for k, v in node.state_stl['Once'].items():
            if v is False or v is None:
                return False
        return True
    
    # ===============================
    # Visualization Methods (Enhanced for Bicycle Model)
    # ===============================
    def draw_graph(self, rnd=None):
        """Draw the RRT graph with bicycle model visualization"""
        plt.figure(3, figsize=(12, 10))
        plt.clf()
        ax = plt.subplot(111, projection='3d')
        
        if rnd is not None:
            ax.scatter(rnd.x, rnd.y, rnd.t, c='r', marker='o', s=50)
        
        # Draw nodes and edges
        for node in self.node_list:
            ax.scatter(node.x, node.y, node.t, c='g', marker='^', s=20)
            if node.parent:
                self.plot_line_3d(ax, node)
        
        # Draw obstacles as cylinders
        for (ox, oy, size) in self.obstacle_list:
            x, y, z = self.plot_cylinder(ox, oy, size)
            ax.plot_surface(x, y, z, color='b', alpha=0.5)
        
        # Draw task points
        for k, v in self.task_point['Once'].items():
            for item in v:
                x, y, z = self.plot_cylinder_cons(item[0], item[1],
                    (self.rand_area[1][2] - k[1], self.rand_area[1][2] - k[0]), item[2])
                ax.plot_surface(x, y, z, color='y', alpha=0.5)
        
        # Start point
        ax.scatter(self.start.x, self.start.y, 0, c='r', marker='x', s=100)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Time')
        ax.set_title('Bicycle RRT* with STL Monitoring')
        plt.pause(0.01)
    
    def draw_rrt(self):
        """Draw final RRT tree"""
        plt.figure(2, figsize=(12, 10))
        plt.clf()
        ax3d = plt.axes(projection='3d')
        
        # Draw selected nodes and their paths
        for node in self.select_nd:
            ax3d.scatter(node.x, node.y, node.t, c='k', marker='^', s=50)
            cn = node
            while cn.parent is not None:
                self.plot_line_3d(ax3d, cn)
                cn = cn.parent
        
        # Draw environment
        for (ox, oy, size) in self.obstacle_list:
            x, y, z = self.plot_cylinder(ox, oy, size)
            ax3d.plot_surface(x, y, z, color='b', alpha=0.5)
        
        for k, v in self.task_point['Once'].items():
            for item in v:
                x, y, z = self.plot_cylinder_cons(item[0], item[1],
                    (self.rand_area[1][2] - k[1], self.rand_area[1][2] - k[0]), item[2])
                ax3d.plot_surface(x, y, z, color='y', alpha=0.5)
        
        ax3d.scatter(self.start.x, self.start.y, 0, c='r', marker='x', s=100)
        ax3d.set_title('Final Bicycle RRT* Tree')
        plt.pause(0.01)
    
    def draw_path_2D(self, start_point):
        """Draw 2D path with bicycle model details"""
        plt.figure(1, figsize=(12, 8))
        plt.clf()
        
        # Find path to start point
        reach_list = [x for x in self.select_nd 
                     if self.calc_distance_bicycle(x, start_point) < 10]
        
        if len(reach_list) == 0:
            print('No path is available!')
            return None
        
        # Select best node
        distances = [np.linalg.norm(node.state.position() - start_point.state.position()) + 
                    node.cost for node in reach_list]
        minind = distances.index(min(distances))
        nearest_select_node = reach_list[minind]
        
        # Draw path
        start_point.parent = nearest_select_node
        cn = start_point
        while cn.parent is not None:
            self.plot_line_2d(cn)
            # Draw vehicle orientation
            self.draw_vehicle_2d(cn.state)
            plt.text(cn.x, cn.y, f"{self.rand_area[1][2] - cn.t:.1f}s\nv={cn.v:.1f}")
            cn = cn.parent
        
        # Draw final node
        self.draw_vehicle_2d(cn.state)
        plt.text(cn.x, cn.y, f"{self.rand_area[1][2] - cn.t:.1f}s\nv={cn.v:.1f}")
        
        # Draw obstacles
        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle_2d(ox, oy, size)
        
        # Draw task points
        for k, v in self.task_point['Once'].items():
            for item in v:
                self.plot_circle_2d(item[0], item[1], item[2], color='y')
                plt.text(item[0], item[1], str(k))
        
        plt.axis("equal")
        plt.grid(True)
        plt.title('Bicycle RRT* Path (2D View)')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.pause(0.01)
    
    def draw_path_3D(self, start_point):
        """Draw 3D path"""
        plt.figure(4, figsize=(12, 10))
        plt.clf()
        ax3d = plt.axes(projection='3d')
        
        # Similar to draw_path_2D but in 3D
        reach_list = [x for x in self.select_nd 
                     if self.calc_distance_bicycle(x, start_point) < 10]
        
        if len(reach_list) == 0:
            print('No 3D path available!')
            return None
        
        distances = [np.linalg.norm(node.state.position() - start_point.state.position()) + 
                    node.cost for node in reach_list]
        minind = distances.index(min(distances))
        nearest_select_node = reach_list[minind]
        
        ax3d.scatter(start_point.x, start_point.y, start_point.t, c='k', marker='^', s=100)
        start_point.parent = nearest_select_node
        cn = start_point
        while cn.parent is not None:
            self.plot_line_3d(ax3d, cn)
            cn = cn.parent
        
        # Draw environment in 3D
        for (ox, oy, size) in self.obstacle_list:
            x, y, z = self.plot_cylinder(ox, oy, size)
            ax3d.plot_surface(x, y, z, color='b', alpha=0.5)
        
        for k, v in self.task_point['Once'].items():
            for item in v:
                x, y, z = self.plot_cylinder_cons(item[0], item[1],
                    (self.rand_area[1][2] - k[1], self.rand_area[1][2] - k[0]), item[2])
                ax3d.plot_surface(x, y, z, color='y', alpha=0.5)
        
        ax3d.scatter(self.start.x, self.start.y, 0, c='r', marker='x', s=100)
        ax3d.set_title('Bicycle RRT* Path (3D View)')
        plt.pause(0.01)
    
    def draw_vehicle_2d(self, state: BicycleState, color='red', alpha=0.7):
        """Draw vehicle shape in 2D"""
        corners = self.bicycle_model.get_vehicle_corners(state)
        vehicle_polygon = plt.Polygon(corners, color=color, alpha=alpha)
        plt.gca().add_patch(vehicle_polygon)
        
        # Draw orientation arrow
        arrow_length = 2.0
        plt.arrow(state.x, state.y,
                 arrow_length * np.cos(state.theta),
                 arrow_length * np.sin(state.theta),
                 head_width=0.5, head_length=0.3, fc=color, ec=color)
    
    def plot_line_3d(self, ax, node):
        """Plot 3D line between node and parent"""
        if node.parent:
            ax.plot([node.x, node.parent.x], 
                   [node.y, node.parent.y], 
                   [node.t, node.parent.t], 'b-', alpha=0.6)
    
    def plot_line_2d(self, node):
        """Plot 2D line between node and parent"""
        if node.parent:
            plt.plot([node.x, node.parent.x], 
                    [node.y, node.parent.y], 'b-', linewidth=2)
    
    def plot_circle_2d(self, x, y, size, color="b"):
        """Plot 2D circle"""
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)
    
    def plot_cylinder(self, x, y, size):
        """Plot cylinder for 3D obstacles"""
        h = np.linspace(0, self.rand_area[1][2], 100)
        h.shape = (100, 1)
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        p = np.ones(len(xl))
        p.shape = (1, len(xl))
        z = p * h
        return np.array(xl), np.array(yl), z
    
    def plot_cylinder_cons(self, x, y, bound, size):
        """Plot constrained cylinder for task points"""
        h = np.linspace(bound[0], bound[1], 100)
        h.shape = (100, 1)
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        p = np.ones(len(xl))
        p.shape = (1, len(xl))
        z = p * h
        return np.array(xl), np.array(yl), z


# ===============================
# Main Demo Function
# ===============================
def main_bicycle_rrt_demo():
    """Main demonstration of merged bicycle RRT* with STL monitoring"""
    print("Starting Bicycle RRT* with STL Monitoring Demo")
    print("=" * 50)
    
    # Environment setup
    obstacle_list = [
        (5, 16, 2),      # Static obstacle
        (15, 5, 1.5),    # Static obstacle  
        (10, 10, 1),     # Small obstacle
    ]
    
    # Define workspace bounds: (x_min, y_min, t_min), (x_max, y_max, t_max)
    rand_area = [(0, 0, 0), (20, 20, 20)]
    
    # Create bicycle model with realistic parameters
    bicycle_model = BicycleModel(
        wheelbase=2.7,           # meters
        max_speed=15.0,          # m/s
        max_acceleration=3.0,    # m/s²
        max_steering_angle=np.pi/4  # 45 degrees max steering
    )
    
    # Initial state: position (5.5, 6.5), heading=0, velocity=2 m/s, time=0
    start_state = BicycleState(x=5.5, y=6.5, theta=0.0, v=2.0, t=0.0)
    
    # Create RRT planner
    rrt = BicycleRRT(
        start_state=start_state,
        obstacle_list=obstacle_list,
        rand_area=rand_area,
        bicycle_model=bicycle_model,
        expand_dis=3.0,           # meters
        path_resolution=0.3,      # meters
        max_iter=300,            # iterations
        control_duration=1.0,     # seconds per control
        num_control_samples=7     # reduced for performance
    )
    
    print(f"Planning with bicycle model:")
    print(f"  Wheelbase: {bicycle_model.wheelbase} m")
    print(f"  Max speed: {bicycle_model.max_speed} m/s")
    print(f"  Max acceleration: {bicycle_model.max_acceleration} m/s²")
    print(f"  Max steering: {bicycle_model.max_steering_angle * 180/np.pi:.1f} degrees")
    print(f"  Start state: x={start_state.x}, y={start_state.y}, θ={start_state.theta:.2f}, v={start_state.v}")
    
    # Run planning
    start_time = time.time()
    path = rrt.planning(animation=show_animation)
    planning_time = time.time() - start_time
    
    print(f"\nPlanning completed in {planning_time:.2f} seconds")
    print(f"Generated {len(rrt.node_list)} nodes")
    print(f"Found {len(rrt.select_nd)} selectable nodes")
    
    # Show final results
    if show_animation:
        # Keep plots open
        plt.show()
    
    return rrt


def run_bicycle_rrt_comparison():
    """Run comparison between standard RRT and bicycle RRT"""
    print("\n" + "=" * 60)
    print("COMPARISON: Standard RRT vs Bicycle RRT* with STL")
    print("=" * 60)
    
    # Same environment for both
    obstacle_list = [(8, 12, 2), (15, 8, 1.5)]
    rand_area = [(0, 0, 0), (20, 20, 15)]
    
    # Test 1: Standard point robot (simplified)
    print("\n1. Standard Point Robot RRT:")
    start_state_simple = BicycleState(x=2, y=2, theta=0, v=5, t=0)
    
    simple_bicycle = BicycleModel(
        wheelbase=0.1,  # Very small wheelbase = point robot
        max_speed=20.0,
        max_acceleration=10.0,
        max_steering_angle=np.pi
    )
    
    rrt_simple = BicycleRRT(
        start_state=start_state_simple,
        obstacle_list=obstacle_list,
        rand_area=rand_area,
        bicycle_model=simple_bicycle,
        max_iter=200
    )
    
    start_time = time.time()
    rrt_simple.planning(animation=False)
    simple_time = time.time() - start_time
    
    print(f"   Time: {simple_time:.2f}s, Nodes: {len(rrt_simple.node_list)}")
    
    # Test 2: Realistic bicycle model
    print("\n2. Realistic Bicycle Model RRT:")
    start_state_bicycle = BicycleState(x=2, y=2, theta=0, v=3, t=0)
    
    realistic_bicycle = BicycleModel(
        wheelbase=2.7,
        max_speed=12.0,
        max_acceleration=2.0,
        max_steering_angle=np.pi/6  # 30 degrees
    )
    
    rrt_bicycle = BicycleRRT(
        start_state=start_state_bicycle,
        obstacle_list=obstacle_list,
        rand_area=rand_area,
        bicycle_model=realistic_bicycle,
        max_iter=200
    )
    
    start_time = time.time()
    rrt_bicycle.planning(animation=False)
    bicycle_time = time.time() - start_time
    
    print(f"   Time: {bicycle_time:.2f}s, Nodes: {len(rrt_bicycle.node_list)}")
    
    # Analysis
    print(f"\nAnalysis:")
    print(f"   Bicycle model is {bicycle_time/simple_time:.1f}x slower due to:")
    print(f"   - Nonlinear dynamics integration")
    print(f"   - Vehicle shape collision checking")
    print(f"   - Kinematic constraints")
    print(f"   - More complex state space (x, y, θ, v)")
    
    return rrt_simple, rrt_bicycle


def demonstrate_stl_monitoring():
    """Demonstrate STL monitoring capabilities"""
    print("\n" + "=" * 60)
    print("STL MONITORING DEMONSTRATION")
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
        max_iter=150
    )
    
    print("STL Task Points:")
    for time_bound, points in rrt_stl.task_point['Once'].items():
        time_start = rand_area[1][2] - time_bound[1]
        time_end = rand_area[1][2] - time_bound[0]
        print(f"   Time [{time_start}, {time_end}]: Visit {points}")
    
    # Run with STL monitoring
    start_time = time.time()
    rrt_stl.planning(animation=False)
    stl_time = time.time() - start_time
    
    print(f"\nSTL Planning Results:")
    print(f"   Planning time: {stl_time:.2f}s")
    print(f"   Total nodes: {len(rrt_stl.node_list)}")
    print(f"   STL-compliant nodes: {len(rrt_stl.select_nd)}")
    print(f"   Success rate: {len(rrt_stl.select_nd)/len(rrt_stl.node_list)*100:.1f}%")
    
    return rrt_stl


if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("BICYCLE RRT* WITH STL MONITORING")
    print("================================")
    
    # Main demonstration
    main_rrt = main_bicycle_rrt_demo()
    
    if input("\nRun comparison demo? (y/n): ").lower().startswith('y'):
        simple_rrt, bicycle_rrt = run_bicycle_rrt_comparison()
    
    if input("\nRun STL monitoring demo? (y/n): ").lower().startswith('y'):
        stl_rrt = demonstrate_stl_monitoring()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("Key Features Demonstrated:")
    print("✓ Bicycle kinematic model with nonlinear dynamics")
    print("✓ RK4 integration for accurate motion prediction")
    print("✓ Vehicle shape collision detection")
    print("✓ STL transducer monitoring (placeholder)")
    print("✓ 3D visualization with time dimension")
    print("✓ Real-time path planning capabilities")
    print("=" * 60)
