"""
utilities/collision.py

Contains helper functions for boundary and collision checks

The map is stored as a 2D grid where every non-zero cell is an obstacle, the value (if non-zero) is used for graphs
"""

import numpy as np

# How many point tests have been made since the last reset. This is the budget a planner is charged: every cell it asks about costs one, whether it asked directly or through a segment. The runner resets it before a run and reads it after, so the number is the planner's own and not the scorer's
_checks = 0


def reset_checks():
    """Set the check counter back to zero, done before each planner run."""
    global _checks
    _checks = 0


def checks_done():
    """Return the number of point tests since the last reset."""
    return _checks


def in_bounds(grid, x, y):
    """Return True if (x, y) is inside bounds"""
    # Gets the bounds
    h, w = grid.shape
    return 0 <= x < w and 0 <= y < h


def point_blocked(grid, x, y):
    """
    Return True if the point (x,y) is either an obstacle or out of bounds
    """
    # One more cell asked about
    global _checks
    _checks += 1

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

    # Finds the number of samples along the segment, at least two points (start and end point). The gaps between samples are one fewer than the samples themselves, so rounding the division up and adding one is what keeps the real gap at or below the requested spacing. Taking the floor instead would sample a short segment at its two endpoints only and step straight over a wall one cell thick
    n_samples = max(2, int(np.ceil(segment_length / spacing)) + 1)

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


def validate_path(grid, path):
    """
    Return True if the whole path is driveable, meaning every point sits on a free cell and every segment between consecutive points stays clear.

    The smoothers call this before they accept a replacement curve, so a curve that cuts a corner through a building is thrown away and the original sharp corner is kept.
    """
    # An empty path or a single point has nothing to drive along
    if path is None or len(path) < 1:
        return False

    # Every point on its own must sit on a free in-bounds cell
    for point in path:
        if point_blocked(grid, point[0], point[1]):
            return False

    # The straight lines joining the points must also be clear
    if path_blocked(grid, path):
        return False

    return True
