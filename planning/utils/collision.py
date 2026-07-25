"""
utilities/collision.py
==================


Contains helper functions for boundary and collision checks

The map is stored as a 2D grid where every non-zero cell is an obstacle, the value (if non-zero) is used for graphs
"""

import numpy as np


def in_bounds(grid, x, y):
    """Return True if (x, y) is inside bounds"""
    # Gets the bounds
    h, w = grid.shape
    return 0 <= x < w and 0 <= y < h


def point_blocked(grid, x, y):
    """
    Return True if the point (x,y) is either an obstacle or out of bounds
    """
    # Rounded to integer since the map array is indexed with integers
    x = int(round(x))
    y = int(round(y))

    # If the point is out of bounds we mark it as blocked
    if not in_bounds(grid, x, y):
        return True
    # If the point is in bounds and non-zero it is an obstacle
    return grid[y, x] != 0


def segment_blocked(grid, point0, point1, spacing=0.5):
    """
    If the straight line between point 0 and point 1 passes through an obstacle or out of bounds it is considered blocked

    Since the line can't be checked infinitely along its points, it is sampled every configured spacing distance and stepped through in increments to sample each point and then check that point.
    """

    # Convert the points to numpy arrays of type float
    point0 = np.asarray(point0, dtype=float)
    point1 = np.asarray(point1, dtype=float)

    # Length of segment connecting points
    segment_length = np.linalg.norm(point1 - point0)

    # Finds the number of samples along the segment, at least two points (start and end point)
    n_samples = max(2, int(segment_length / spacing))

    # Finds evenly-spaced fractions between 0 and 1. The number of fractions is the number of samples, so at each fraction we take a sample
    for fraction in np.linspace(0.0, 1.0, n_samples):
        # Find the sample point by going fraction across the segment
        point = point0 + fraction * (point1 - point0)

        # If the point is blocked then the segment is blocked at that point
        if point_blocked(grid, point[0], point[1]):
            return True
    return False


def path_blocked(grid, path):
    """
    Checks if any segment of the path goes out of bounds or through an obstacle by checking the segment between consecutive points in the path.
    """
    # For each point in the path
    for i in range(len(path) - 1):
        # Check if the segment between the point and next consecutive point is blocked (path from one point to another), prove the path is valid using inductive analysis
        if segment_blocked(grid, path[i], path[i + 1]):
            return True
    return False
