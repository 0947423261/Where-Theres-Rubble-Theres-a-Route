"""
algorithms/base_planner.py

==============
Abstract class containing methods to be implemented by the path planning algorithms to allow for modular design and easy extendability with less boilerplate
"""

from abc import ABC, abstractmethod
import numpy as np
from ..utils.collision import point_blocked, segment_blocked, path_blocked


class BasePlanner(ABC):
    # Constructor consisting of the grid, start, goal and configuration
    def __init__(self, grid, start, goal, config, rng):
        self.grid = grid
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.config = config
        self.rng = rng

    @abstractmethod
    def plan(self):
        """
        Run the planning algorithm and return a dictionary with the path from start to finish, whether the algorithm succeeded and the number of iterations.
        """
        raise NotImplementedError

    def _arrived(self, pos):
        # normalises the difference vector from current position to goal and checks if we are within tolerance
        return np.linalg.norm(pos - self.goal) <= self.config.goal_tolerance

    def _point_blocked(self, pos):
        return point_blocked(self.grid, pos[0], pos[1])

    def _segment_blocked(self, a, b):
        return segment_blocked(self.grid, a, b)

    def _validate_path(self, path):
        """Check that all points along a path are free and all segments are collision‑free."""
        # If the path is empty return false
        if not path:
            return False
        # For every point in the path check if it is blocked
        for point in path:
            if self._point_blocked(point):
                return False
        # Check if any part of the path goes through an obstacle
        if path_blocked(self.grid, path):
            return False
        return True
