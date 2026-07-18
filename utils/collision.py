"""
utilities/collision.py
===============
File containing code to find all types of collisions and boundary checks

The map is stored in a 2D array where cells with 0 mean the cell is free otherwise the cell has an obstacle (rubble or building). The difference in value for rubble and buildings is for animations.

For reference positions are written [x, y]  but the array is indexed grid[y, x]
"""

import numpy as np


def in_bounds(grid, x, y):
    """Return True if (x, y) is inside the grid."""
    h, w = grid.shape
    return 0 <= x < w and 0 <= y < h


def point_blocked(grid, x, y):
    """
    Return True if the point is off the edge of the map or is an obstacle. The position is rounded to nearest integer since the grid uses integer indexes.
    """
    xi = int(round(x))
    yi = int(round(y))

    # If not in bounds it is blocked
    if not in_bounds(grid, xi, yi):
        return True
    # Return True if there is obstacle otherwise False
    return grid[yi, xi] != 0


def segment_blocked(grid, point_0, point_1, spacing=0.5):
    """
    Return True if the straight line from point point_0 to point point_1 passes through
    ANY obstacle.

    Since it is not feasible to check infinitely across the line, step along and check each sample point.
    """

    # Convert both points to numpy arrays of type float
    p0 = np.asarray(point_0, dtype=float)
    p1 = np.asarray(point_1, dtype=float)

    # Find the magnitude of the line between both points
    dist = np.linalg.norm(p1 - p0)
    # Find the number of samples (at minimum 2)
    n_samples = max(2, int(dist / spacing))

    # Splits 1 into n_samples fractions
    for t in np.linspace(0.0, 1.0, n_samples):
        # Finds the point that is the sample's corresponding fraction across the line
        point = p0 + t * (p1 - p0)
        # Checks if the sample is blocked
        if point_blocked(grid, point[0], point[1]):
            return True
    return False


def path_blocked(grid, path):
    """
    Return True if ANY segment of a whole path collides with an obstacle.
    """
    # For every point in the path check that the point and the consecutive point are not blocked by an obstacle
    for i in range(len(path) - 1):
        if segment_blocked(grid, path[i], path[i + 1]):
            return True
    return False
