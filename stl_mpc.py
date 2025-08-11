import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import time
import random
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
import math

@dataclass
class BicycleState:
    """State representation for bicycle model: [x, y, theta, v]"""
    x: float = 0.0      # x position
    y: float = 0.0      # y position
    theta: float = 0.0  # heading angle
    v: float = 0.0      # velocity
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'BicycleState':
        return cls(arr[0], arr[1], arr[2], arr[3])
    
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
    """Bicycle kinematic model"""
    
    def __init__(self, wheelbase: float = 2.7, max_speed: float = 30.0, 
                 max_acceleration: float = 5.0, max_steering_angle: float = np.pi/4):
        self.wheelbase = wheelbase  # distance between front and rear axles
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
        k2 = self.dynamics(BicycleState.from_array(state_vec + 0.5 * dt * k1), control)
        k3 = self.dynamics(BicycleState.from_array(state_vec + 0.5 * dt * k2), control)
        k4 = self.dynamics(BicycleState.from_array(state_vec + dt * k3), control)
        
        new_state_vec = state_vec + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Clamp velocity
        new_state_vec[3] = np.clip(new_state_vec[3], 0, self.max_speed)
        
        # Normalize angle
        new_state_vec[2] = self.normalize_angle(new_state_vec[2])
        
        return BicycleState.from_array(new_state_vec)
    
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

@dataclass
class Node:
    """A node in the RT-RRT* tree for bicycle model"""
    state: BicycleState
    parent: Optional['Node'] = None
    children: Set['Node'] = field(default_factory=set)
    cost: float = 0.0
    control_from_parent: Optional[BicycleControl] = None
    integration_time: float = 0.0
    
    def __hash__(self):
        return id(self)
    
    def __eq__(self, other):
        return id(self) == id(other)

class ObstacleManager:
    """Manages dynamic obstacles in the environment"""
    
    def __init__(self):
        self.obstacles = []  # List of (center, radius, velocity) tuples
        
    def add_obstacle(self, center: np.ndarray, radius: float, velocity: np.ndarray = None):
        center = np.array(center, dtype=np.float64)  # force float64
        if velocity is None:
            velocity = np.zeros(2, dtype=np.float64)
        else:
            velocity = np.array(velocity, dtype=np.float64)
        self.obstacles.append([center.copy(), radius, velocity.copy()])

    
    def update_obstacles(self, dt: float):
        """Update obstacle positions based on their velocities"""
        for obstacle in self.obstacles:
            obstacle[0] += obstacle[2] * dt  # center += velocity * dt
    
    def is_state_collision(self, state: BicycleState, vehicle_model: BicycleModel, 
                          safety_margin: float = 0.0) -> bool:
        """Check if a vehicle state collides with any obstacle"""
        corners = vehicle_model.get_vehicle_corners(state)
        
        for center, radius, _ in self.obstacles:
            # Check each corner of the vehicle
            for corner in corners:
                if np.linalg.norm(corner - center) <= radius + safety_margin:
                    return True
            
            # Also check if obstacle center is inside vehicle (additional safety)
            vehicle_center = state.position()
            if np.linalg.norm(vehicle_center - center) <= radius + safety_margin:
                return True
                
        return False
    
    def is_trajectory_collision_free(self, start_state: BicycleState, control: BicycleControl,
                                   duration: float, vehicle_model: BicycleModel,
                                   safety_margin: float = 0.0, num_checks: int = 10) -> bool:
        """Check if a trajectory is collision-free"""
        dt = duration / num_checks
        current_state = start_state
        
        for i in range(num_checks + 1):
            if self.is_state_collision(current_state, vehicle_model, safety_margin):
                return False
            if i < num_checks:
                current_state = vehicle_model.integrate(current_state, control, dt)
        
        return True

class SpatialIndex:
    """Grid-based spatial indexing for fast neighbor queries"""
    
    def __init__(self, bounds: Tuple[float, float, float, float], cell_size: float):
        self.bounds = bounds  # (x_min, x_max, y_min, y_max)
        self.cell_size = cell_size
        self.grid = {}
        
    def _get_cell(self, position: np.ndarray) -> Tuple[int, int]:
        """Get grid cell for a position"""
        x, y = position
        cell_x = int((x - self.bounds[0]) / self.cell_size)
        cell_y = int((y - self.bounds[2]) / self.cell_size)
        return (cell_x, cell_y)
    
    def add_node(self, node: Node):
        """Add a node to the spatial index"""
        cell = self._get_cell(node.state.position())
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(node)
    
    def remove_node(self, node: Node):
        """Remove a node from the spatial index"""
        cell = self._get_cell(node.state.position())
        if cell in self.grid and node in self.grid[cell]:
            self.grid[cell].remove(node)
    
    def get_neighbors(self, position: np.ndarray, radius: float) -> List[Node]:
        """Get all nodes within radius of position"""
        neighbors = []
        cell_radius = int(np.ceil(radius / self.cell_size))
        center_cell = self._get_cell(position)
        
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                if cell in self.grid:
                    for node in self.grid[cell]:
                        if np.linalg.norm(node.state.position() - position) <= radius:
                            neighbors.append(node)
        return neighbors

class BicycleRTRRTStar:
    """Real-Time RRT* implementation for bicycle model"""
    
    def __init__(self, bounds: Tuple[float, float, float, float], 
                 obstacle_manager: ObstacleManager,
                 vehicle_model: BicycleModel,
                 max_nodes: int = 2000,
                 search_radius: float = 30.0,
                 iteration_time: float = 0.1,  # 100ms iterations
                 control_duration: float = 1.0,  # duration for each control input
                 num_control_samples: int = 11):
        
        self.bounds = bounds  # (x_min, x_max, y_min, y_max)
        self.obstacle_manager = obstacle_manager
        self.vehicle_model = vehicle_model
        self.max_nodes = max_nodes
        self.search_radius = search_radius
        self.iteration_time = iteration_time
        self.control_duration = control_duration
        self.num_control_samples = num_control_samples
        
        # Tree structure
        self.nodes = []
        self.root = None
        
        # Spatial indexing for fast neighbor queries
        cell_size = self.search_radius / 2
        self.spatial_index = SpatialIndex(bounds, cell_size)
        
        # Planning state
        self.goal_state = None
        self.current_plan = []
        
        # Control sampling parameters
        self.control_samples = self._generate_control_samples()
        
    def _generate_control_samples(self) -> List[BicycleControl]:
        """Generate a set of control samples for exploration"""
        controls = []
        
        # Acceleration samples
        acc_samples = np.linspace(-self.vehicle_model.max_acceleration, 
                                 self.vehicle_model.max_acceleration, 
                                 self.num_control_samples)
        
        # Steering angle samples
        steer_samples = np.linspace(-self.vehicle_model.max_steering_angle,
                                   self.vehicle_model.max_steering_angle,
                                   self.num_control_samples)
        
        # Combine acceleration and steering samples
        for acc in acc_samples:
            for steer in steer_samples:
                controls.append(BicycleControl(acc, steer))
        
        return controls
        
    def initialize_tree(self, start_state: BicycleState):
        """Initialize the tree with a root node"""
        self.root = Node(state=start_state)
        self.nodes = [self.root]
        self.spatial_index.add_node(self.root)
        
    def reroot_tree(self, new_root_state: BicycleState):
        """Reroot the tree to a new state (robot's current state)"""
        if self.root is None:
            self.initialize_tree(new_root_state)
            return
            
        # Find the closest node to the new root state
        closest_node = self.find_nearest_node(new_root_state)
        
        # If the closest node is close enough, make it the new root
        pos_diff = np.linalg.norm(closest_node.state.position() - new_root_state.position())
        angle_diff = abs(BicycleModel.normalize_angle(closest_node.state.theta - new_root_state.theta))
        
        if pos_diff < 5.0 and angle_diff < np.pi/6:  # 5 units position, 30 deg angle tolerance
            self._reroot_to_node(closest_node)
        else:
            # Create a new root node
            new_root = Node(state=new_root_state)
            # Don't connect to existing tree immediately - let rewiring handle it
            self.nodes.append(new_root)
            self.spatial_index.add_node(new_root)
            self._reroot_to_node(new_root)
    
    def _reroot_to_node(self, new_root: Node):
        """Reroot the tree to an existing node"""
        if new_root.parent is not None:
            # Reverse the path from new_root to current root
            path_to_reverse = []
            current = new_root
            while current.parent is not None:
                path_to_reverse.append((current, current.parent))
                current = current.parent
            
            # Reverse parent-child relationships
            for child, parent in path_to_reverse:
                parent.children.remove(child)
                child.children.add(parent)
                parent.parent = child
                child.parent = None
                
        self.root = new_root
        self.root.parent = None
        self._update_costs_recursive(self.root, 0.0)
    
    def _update_costs_recursive(self, node: Node, cost: float):
        """Recursively update costs from a node"""
        node.cost = cost
        for child in node.children:
            # Cost includes time and distance components
            time_cost = child.integration_time if child.integration_time > 0 else self.control_duration
            distance_cost = np.linalg.norm(child.state.position() - node.state.position())
            edge_cost = time_cost + 0.1 * distance_cost  # Weight time more heavily
            self._update_costs_recursive(child, cost + edge_cost)
    
    def find_nearest_node(self, target_state: BicycleState) -> Node:
        """Find the nearest node to a given state"""
        min_cost = float('inf')
        nearest_node = None
        
        for node in self.nodes:
            # Distance metric combining position and orientation
            pos_diff = np.linalg.norm(node.state.position() - target_state.position())
            angle_diff = abs(BicycleModel.normalize_angle(node.state.theta - target_state.theta))
            vel_diff = abs(node.state.v - target_state.v)
            
            # Weighted distance metric
            cost = pos_diff + 10.0 * angle_diff + 2.0 * vel_diff
            
            if cost < min_cost:
                min_cost = cost
                nearest_node = node
                
        return nearest_node
    
    def sample_random_state(self) -> BicycleState:
        """Sample a random state in the workspace"""
        x = random.uniform(self.bounds[0], self.bounds[1])
        y = random.uniform(self.bounds[2], self.bounds[3])
        theta = random.uniform(-np.pi, np.pi)
        v = random.uniform(0, self.vehicle_model.max_speed)
        return BicycleState(x, y, theta, v)
    
    def steer(self, from_state: BicycleState, to_state: BicycleState) -> Tuple[BicycleControl, float]:
        """Find the best control to steer from one state towards another"""
        best_control = None
        best_distance = float('inf')
        best_duration = self.control_duration
        
        # Try different control durations
        for duration in [0.5, 1.0, 1.5]:
            for control in self.control_samples:
                # Simulate forward
                final_state = self.vehicle_model.integrate(from_state, control, duration)
                
                # Calculate distance to target
                pos_diff = np.linalg.norm(final_state.position() - to_state.position())
                angle_diff = abs(BicycleModel.normalize_angle(final_state.theta - to_state.theta))
                vel_diff = abs(final_state.v - to_state.v)
                
                distance = pos_diff + 10.0 * angle_diff + 2.0 * vel_diff
                
                if distance < best_distance:
                    best_distance = distance
                    best_control = control
                    best_duration = duration
        
        return best_control, best_duration
    
    def extend_tree(self, time_budget: float):
        """Extend the tree within the given time budget"""
        start_time = time.time()
        
        while time.time() - start_time < time_budget and len(self.nodes) < self.max_nodes:
            # Sample a random state
            if self.goal_state is not None and random.random() < 0.1:
                # 10% chance to sample towards goal
                random_state = self.goal_state
            else:
                random_state = self.sample_random_state()
            
            # Find nearest node
            nearest_node = self.find_nearest_node(random_state)
            if nearest_node is None:
                continue
                
            # Steer towards the random state
            control, duration = self.steer(nearest_node.state, random_state)
            if control is None:
                continue
            
            # Simulate forward to get new state
            new_state = self.vehicle_model.integrate(nearest_node.state, control, duration)
            
            # Check if the new trajectory is collision-free
            if not self.obstacle_manager.is_trajectory_collision_free(
                nearest_node.state, control, duration, self.vehicle_model):
                continue
            
            # Create new node
            new_node = Node(
                state=new_state,
                control_from_parent=control,
                integration_time=duration
            )
            
            # Find neighbors for rewiring
            neighbors = self.spatial_index.get_neighbors(new_state.position(), self.search_radius)
            
            # Choose parent (RRT* optimization)
            best_parent = nearest_node
            time_cost = duration
            distance_cost = np.linalg.norm(new_state.position() - nearest_node.state.position())
            best_cost = nearest_node.cost + time_cost + 0.1 * distance_cost
            
            for neighbor in neighbors:
                if neighbor != nearest_node:
                    # Try to connect from neighbor to new_state
                    test_control, test_duration = self.steer(neighbor.state, new_state)
                    if test_control is not None:
                        # Check if trajectory is collision-free
                        if self.obstacle_manager.is_trajectory_collision_free(
                            neighbor.state, test_control, test_duration, self.vehicle_model):
                            
                            time_cost = test_duration
                            distance_cost = np.linalg.norm(new_state.position() - neighbor.state.position())
                            cost = neighbor.cost + time_cost + 0.1 * distance_cost
                            
                            if cost < best_cost:
                                best_cost = cost
                                best_parent = neighbor
                                new_node.control_from_parent = test_control
                                new_node.integration_time = test_duration
            
            # Add new node to tree
            new_node.parent = best_parent
            new_node.cost = best_cost
            best_parent.children.add(new_node)
            self.nodes.append(new_node)
            self.spatial_index.add_node(new_node)
            
            # Rewire neighbors (simplified for time constraints)
            for neighbor in neighbors[:5]:  # Limit rewiring for real-time performance
                if neighbor != best_parent and neighbor.parent is not None:
                    # Try to connect from new_node to neighbor
                    test_control, test_duration = self.steer(new_state, neighbor.state)
                    if test_control is not None:
                        time_cost = test_duration
                        distance_cost = np.linalg.norm(neighbor.state.position() - new_state.position())
                        new_cost = new_node.cost + time_cost + 0.1 * distance_cost
                        
                        if (new_cost < neighbor.cost and 
                            self.obstacle_manager.is_trajectory_collision_free(
                                new_state, test_control, test_duration, self.vehicle_model)):
                            
                            # Remove old parent connection
                            neighbor.parent.children.remove(neighbor)
                            
                            # Set new parent
                            neighbor.parent = new_node
                            neighbor.control_from_parent = test_control
                            neighbor.integration_time = test_duration
                            new_node.children.add(neighbor)
                            
                            # Update costs
                            self._update_costs_recursive(neighbor, new_cost)
    
    def rewire_tree(self, time_budget: float):
        """Rewire the tree within the given time budget"""
        start_time = time.time()
        
        # Process nodes starting from root
        nodes_to_process = [self.root] if self.root else []
        processed = set()
        
        while time.time() - start_time < time_budget and nodes_to_process:
            current_node = nodes_to_process.pop(0)
            if current_node in processed:
                continue
                
            processed.add(current_node)
            nodes_to_process.extend(list(current_node.children)[:3])  # Limit children for real-time
            
            # Skip rewiring for root
            if current_node == self.root:
                continue
            
            # Find neighbors for potential rewiring
            neighbors = self.spatial_index.get_neighbors(
                current_node.state.position(), self.search_radius)
            
            # Try to find a better parent
            best_parent = current_node.parent
            best_cost = current_node.cost
            best_control = current_node.control_from_parent
            best_duration = current_node.integration_time
            
            for neighbor in neighbors[:5]:  # Limit for real-time performance
                if (neighbor != current_node and neighbor not in current_node.children):
                    
                    # Try to connect from neighbor to current_node
                    test_control, test_duration = self.steer(neighbor.state, current_node.state)
                    if test_control is not None:
                        if self.obstacle_manager.is_trajectory_collision_free(
                            neighbor.state, test_control, test_duration, self.vehicle_model):
                            
                            time_cost = test_duration
                            distance_cost = np.linalg.norm(
                                current_node.state.position() - neighbor.state.position())
                            new_cost = neighbor.cost + time_cost + 0.1 * distance_cost
                            
                            if new_cost < best_cost:
                                best_cost = new_cost
                                best_parent = neighbor
                                best_control = test_control
                                best_duration = test_duration
            
            # Rewire if better parent found
            if best_parent != current_node.parent:
                # Remove old connection
                current_node.parent.children.remove(current_node)
                
                # Create new connection
                current_node.parent = best_parent
                current_node.control_from_parent = best_control
                current_node.integration_time = best_duration
                best_parent.children.add(current_node)
                
                # Update costs
                self._update_costs_recursive(current_node, best_cost)
    
    def update_for_dynamic_obstacles(self):
        """Update tree structure for dynamic obstacles"""
        # Remove nodes that are now in collision
        nodes_to_remove = []
        for node in self.nodes:
            if self.obstacle_manager.is_state_collision(node.state, self.vehicle_model):
                nodes_to_remove.append(node)
        
        for node in nodes_to_remove:
            self._remove_node_and_descendants(node)
        
        # Update costs for remaining nodes
        if self.root is not None:
            self._update_costs_recursive(self.root, 0.0)
    
    def _remove_node_and_descendants(self, node: Node):
        """Remove a node and all its descendants from the tree"""
        # Recursively remove all descendants
        for child in list(node.children):
            self._remove_node_and_descendants(child)
        
        # Remove from parent's children
        if node.parent is not None:
            node.parent.children.remove(node)
        
        # Remove from tree structures
        if node in self.nodes:
            self.nodes.remove(node)
        self.spatial_index.remove_node(node)
        
        # If removing root, find new root
        if node == self.root:
            self.root = self.nodes[0] if self.nodes else None
    
    def plan_to_goal(self, goal_state: BicycleState) -> List[Tuple[BicycleState, BicycleControl, float]]:
        """Plan a path to the goal state"""
        self.goal_state = goal_state
        
        # Find nearest node to goal
        if not self.nodes:
            return []
            
        nearest_to_goal = self.find_nearest_node(goal_state)
        
        # Try to connect directly to goal
        control, duration = self.steer(nearest_to_goal.state, goal_state)
        
        if (control is not None and 
            self.obstacle_manager.is_trajectory_collision_free(
                nearest_to_goal.state, control, duration, self.vehicle_model)):
            
            # Build path with controls
            path = []
            current = nearest_to_goal
            while current.parent is not None:
                path.append((current.state, current.control_from_parent, current.integration_time))
                current = current.parent
            
            # Add root state
            path.append((current.state, None, 0.0))
            
            # Add goal connection
            path.insert(0, (goal_state, control, duration))
            
            path.reverse()
            return path
        else:
            # Return path to nearest feasible node
            path = []
            current = nearest_to_goal
            while current.parent is not None:
                path.append((current.state, current.control_from_parent, current.integration_time))
                current = current.parent
            
            # Add root state
            path.append((current.state, None, 0.0))
            
            path.reverse()
            return path
    
    def run_iteration(self, robot_state: BicycleState, goal_state: BicycleState) -> List[Tuple[BicycleState, BicycleControl, float]]:
        """Run one iteration of RT-RRT*"""
        iteration_start = time.time()
        
        # Update obstacle positions
        self.obstacle_manager.update_obstacles(self.iteration_time)
        
        # Reroot tree to robot state
        self.reroot_tree(robot_state)
        
        # Update tree for dynamic obstacles
        self.update_for_dynamic_obstacles()
        
        # Allocate remaining time for tree extension and rewiring
        elapsed = time.time() - iteration_start
        remaining_time = self.iteration_time - elapsed
        
        if remaining_time > 0:
            # Split remaining time between extension and rewiring
            extend_time = remaining_time * 0.6
            rewire_time = remaining_time * 0.4
            
            # Extend tree
            self.extend_tree(extend_time)
            
            # Rewire tree
            self.rewire_tree(rewire_time)
        
        # Plan path to goal
        path = self.plan_to_goal(goal_state)
        
        return path
    
    def get_tree_visualization_data(self):
        """Get data for visualizing the tree"""
        edges = []
        node_positions = []
        node_orientations = []
        
        for node in self.nodes:
            node_positions.append(node.state.position())
            node_orientations.append(node.state.theta)
            if node.parent is not None:
                edges.append([node.parent.state.position(), node.state.position()])
        
        return node_positions, node_orientations, edges

# Example usage and visualization
def run_bicycle_rt_rrt_star_demo():
    """Run a demonstration of RT-RRT* with bicycle model"""
    
    # Environment setup
    bounds = (0, 500, 0, 400)  # (x_min, x_max, y_min, y_max)
    obstacle_manager = ObstacleManager()
    
    # Add a dynamic obstacle (moving human)
    obstacle_center = np.array([250, 350])
    # obstacle_velocity = np.array([0, -55])  # Moving downward at 0.55 m/s
    obstacle_velocity = np.array([0, 0])  # static
    obstacle_manager.add_obstacle(obstacle_center, 25, obstacle_velocity)
    
    # Bicycle model setup
    vehicle_model = BicycleModel(wheelbase=2.7, max_speed=25.0, max_acceleration=3.0)
    
    # Initialize RT-RRT*
    planner = BicycleRTRRTStar(bounds, obstacle_manager, vehicle_model, max_nodes=1000)
    
    # Robot starts at bottom and wants to go to top
    start_state = BicycleState(x=50, y=50, theta=np.pi/4, v=5.0)
    goal_state = BicycleState(x=450, y=350, theta=0, v=10.0)
    
    planner.initialize_tree(start_state)
    
    # Simulation parameters
    simulation_time = 100  # seconds
    dt = 0.1  # time step
    
    # Run simulation
    current_state = start_state
    time_steps = []
    robot_trajectory = []
    
    plt.figure(figsize=(14, 10))
    
    for t in np.arange(0, simulation_time, dt):
        print(f"Time: {t:.1f}s")
        
        # Run RT-RRT* iteration
        path = planner.run_iteration(current_state, goal_state)
        
        # Execute control along path
        if len(path) > 1 and path[1][1] is not None:  # path[1] = (state, control, duration)
            control = path[1][1]
            control_duration = min(path[1][2], dt)  # Don't exceed time step
            current_state = vehicle_model.integrate(current_state, control, control_duration)
        
        # Store data for visualization
        time_steps.append(t)
        robot_trajectory.append([current_state.x, current_state.y])
        
        # Visualization every 0.5 seconds
        if abs(t - round(t * 2) / 2) < 0.05:
            plt.clf()
            
            # Plot environment bounds
            plt.xlim(bounds[0], bounds[1])
            plt.ylim(bounds[2], bounds[3])
            
            # Plot tree
            node_positions, node_orientations, edges = planner.get_tree_visualization_data()
            for edge in edges:
                plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 
                        'g-', alpha=0.3, linewidth=0.5)
            
            if node_positions:
                positions = np.array(node_positions)
                orientations = np.array(node_orientations)
                
                # Plot nodes as small circles
                plt.scatter(positions[:, 0], positions[:, 1], c='green', s=1, alpha=0.5)
                
                # Plot orientation arrows for some nodes
                subsample = slice(0, len(positions), max(1, len(positions)//50))
                plt.quiver(positions[subsample, 0], positions[subsample, 1],
                          np.cos(orientations[subsample]), np.sin(orientations[subsample]),
                          alpha=0.3, scale=20, color='green', width=0.002)
            
            # Plot obstacles
            for center, radius, _ in obstacle_manager.obstacles:
                circle = Circle(center, radius, color='blue', alpha=0.7)
                plt.gca().add_patch(circle)
            
            # Plot current vehicle state
            vehicle_corners = vehicle_model.get_vehicle_corners(current_state)
            vehicle_rect = plt.Polygon(vehicle_corners, color='red', alpha=0.8)
            plt.gca().add_patch(vehicle_rect)
            
            # Plot vehicle orientation arrow
            arrow_length = 8
            plt.arrow(current_state.x, current_state.y,
                     arrow_length * np.cos(current_state.theta),
                     arrow_length * np.sin(current_state.theta),
                     head_width=3, head_length=2, fc='red', ec='red')
            
            # Plot goal state
            goal_corners = vehicle_model.get_vehicle_corners(goal_state)
            goal_rect = plt.Polygon(goal_corners, color='gold', alpha=0.6)
            plt.gca().add_patch(goal_rect)
            
            # Plot goal orientation arrow
            plt.arrow(goal_state.x, goal_state.y,
                     arrow_length * np.cos(goal_state.theta),
                     arrow_length * np.sin(goal_state.theta),
                     head_width=3, head_length=2, fc='gold', ec='gold', alpha=0.6)
            
            # Plot planned path
            if len(path) > 1:
                path_positions = [state_control[0].position() for state_control in path]
                path_array = np.array(path_positions)
                plt.plot(path_array[:, 0], path_array[:, 1], 'r--', linewidth=2, alpha=0.8)
                
                # Show orientations along path
                for i, (state, _, _) in enumerate(path[::3]):  # Every 3rd point
                    plt.arrow(state.x, state.y,
                             5 * np.cos(state.theta), 5 * np.sin(state.theta),
                             head_width=2, head_length=1.5, fc='red', ec='red', alpha=0.6)
            
            # Plot robot trajectory
            if len(robot_trajectory) > 1:
                traj_array = np.array(robot_trajectory)
                plt.plot(traj_array[:, 0], traj_array[:, 1], 'red', linewidth=1, alpha=0.6)
            
            plt.title(f'Bicycle RT-RRT* at t={t:.1f}s\nVelocity: {current_state.v:.1f} m/s, Heading: {np.degrees(current_state.theta):.1f}°')
            plt.xlabel('X (units)')
            plt.ylabel('Y (units)')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.pause(0.1)
        
        # Check if goal reached (position and orientation)
        pos_error = np.linalg.norm(current_state.position() - goal_state.position())
        angle_error = abs(BicycleModel.normalize_angle(current_state.theta - goal_state.theta))
        
        if pos_error < 10 and angle_error < np.pi/6:  # 10 units position, 30 degrees orientation
            print("Goal reached!")
            break
    
    plt.show()
    print(f"Final state: x={current_state.x:.1f}, y={current_state.y:.1f}, theta={np.degrees(current_state.theta):.1f}°, v={current_state.v:.1f}")
    print(f"Distance to goal: {np.linalg.norm(current_state.position() - goal_state.position()):.1f} units")
    print(f"Angle error: {np.degrees(abs(BicycleModel.normalize_angle(current_state.theta - goal_state.theta))):.1f}°")

def test_bicycle_model():
    """Test the bicycle model dynamics"""
    model = BicycleModel()
    
    # Test straight line motion
    state = BicycleState(0, 0, 0, 10)  # 10 m/s forward
    control = BicycleControl(0, 0)  # No acceleration, no steering
    
    print("Testing bicycle model:")
    for i in range(10):
        print(f"t={i:.1f}s: x={state.x:.1f}, y={state.y:.1f}, theta={np.degrees(state.theta):.1f}°, v={state.v:.1f}")
        state = model.integrate(state, control, 0.1)
    
    print("\nTesting turning:")
    state = BicycleState(0, 0, 0, 10)  # 10 m/s forward
    control = BicycleControl(0, np.pi/6)  # 30 degree steering
    
    for i in range(20):
        if i % 5 == 0:
            print(f"t={i*0.1:.1f}s: x={state.x:.1f}, y={state.y:.1f}, theta={np.degrees(state.theta):.1f}°, v={state.v:.1f}")
        state = model.integrate(state, control, 0.1)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Test bicycle model first
    print("=" * 50)
    print("Testing Bicycle Model")
    print("=" * 50)
    test_bicycle_model()
    print()
    
    # Run the main demonstration
    print("=" * 50)
    print("Running Bicycle RT-RRT* Demo")
    print("=" * 50)
    run_bicycle_rt_rrt_star_demo()