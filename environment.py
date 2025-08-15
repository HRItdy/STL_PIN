"""
Environment specification for STL RRT planning

Defines obstacles, workspace bounds, and STL task specification.
"""
import numpy as np

# Obstacles: list of (x, y, radius)
OBSTACLE_LIST = [
    (5, 16, 2),
    # (15, 5, 1.5),
    (10, 10, 1),
]

# Workspace bounds: (x_min, y_min, t_min), (x_max, y_max, t_max)
RAND_AREA = [(0, 0, 0), (20, 20, 20)]

# STL task specification (example: visit region A then region B within time bounds)
# Define region predicates as circles: (center_x, center_y, radius)
REGION_A = (15, 8, 3)
REGION_B = (13, 16, 3)
REGION_C = (5, 6, 3)

STL_FORMULA = "F[0,5] (A) & F[0,10] (B)"

# Region predicate mapping for the transducer
REGION_PREDICATES = {
    'A': REGION_A,
    'B': REGION_B,
    'C': REGION_C,
}