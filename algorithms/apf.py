"""
alorithms/apf.py

Artificial Potential Field (APF) planner as proposed by Khatib, 1986

===============
The goal attracts the vehicle whilst obstacles repel the vehicle. At each step, the resultant force is calculated and a small step is taken in that direction. This is repeated until the goal is reached or a livelock state.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def _sample(field, x, y):
    """Reads the distance value from a point in the 2D plane at the nearest cell to (x,y)"""
    h, w = field.shape
    # Round to the nearest integer coordinates from floating point and ensure the coordinates are not out-of-bounds
    xi = min(max(int(round(x)), 0), w - 1)
    yi = min(max(int(round(y)), 0), h - 1)
    # Index the array to get the distance value
    return field[yi, xi]


def _apf_force(pos, goal, dist_field, grad_x, grad_x, config):
    """
    Computes the resultant vector acting on the vehicle.
    """
    # Converts the current position and goal position into numpy arrays in float type
    pos = np.asarray(pos, dtype=float)
    goal = np.asarray(goal, dtype=float)

    # TODO: complete this function


def _precompute_fields(grid):
    """
    Build the distance field and its gradient once, so every APF step is cheap.
    Returns (dist_field, grad_x, grad_y).
    """
    # grid == 0 means that free space is set to True and obstacles are False. The Scipy function finds the euclidean distance from every False cell.
    # Obstacles have 0 value and cells further away are "higher" and have large value
    dist_field = distance_transform_edt(grid == 0)
    # This calculates the slope at each cell telling you the direction the distance increases value increases the fastest. This is the direction of the repulsive force
    grad_y, grad_x = np.gradient(dist_field)
    return dist_field, grad_x, grad_y


def run_apf(grid, start, goal, config):
    """
    Run APF to compute an optimised path.

    Returns a dictionary with the path (a list of [x,y] points along the route), whether it successfully reached the goal and the number of iterations/steps it took
    """
    dist_field, grad_x, grad_y = _precompute_fields(grid)

    pos = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    # TODO: consider shape

    # If we exit the loop we never reached the goal.
    return {"path": path, "success": False, "iters": config.apf_max_iter}
