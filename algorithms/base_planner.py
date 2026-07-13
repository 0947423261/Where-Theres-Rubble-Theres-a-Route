"""
algorithms/base_planner.py

==============
Abstract class containing methods to be implemented by the path planning algorithms to allow for modular design and easy extendability with less boilerplate
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePlanner(ABC):
    # Constructor consisting of the grid, start, goal and configuration
    def __init__(self, grid, start, goal, config):
        self.grid = grid
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.config = config

    @abstractmethod
    def plan(self):
        """
        Run the planning algorithm and return a dictionary with the path from start to finish, whether the algorithm succeeded and the number of iterations.
        """
        raise NotImplementedError

    def _arrived(self, pos):
        # normalises the difference vector from current position to goal and checks if we are within tolerance
        return np.linalg.norm(pos - self.goal) <= self.config.goal_tolerance
