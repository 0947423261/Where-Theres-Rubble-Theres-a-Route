"""
algorithms/apf.py

Artificial Potential Field (APF) planner as proposed by Khatib, 1986

===============
The goal attracts the vehicle whilst obstacles repel the vehicle. At each step, the resultant force is calculated and a small step is taken in that direction. This is repeated until the goal is reached or a livelock state.
"""

import numpy as np
from .base_planner import BasePlanner
from .apf_mixin import APFMixin


class APF(BasePlanner, APFMixin):
    # Initialise an object of the BasePlanner class to inherit from
    def __init__(self, grid, start, goal, config, rng):
        super().__init__(grid, start, goal, config, rng)

    @staticmethod
    def _sample(field, x, y):
        """Reads the distance value from a point in the 2D plane at the nearest cell to (x,y)"""
        h, w = field.shape
        # Round to the nearest integer coordinates from floating point and ensure the coordinates are not out-of-bounds
        xi = min(max(int(round(x)), 0), w - 1)
        yi = min(max(int(round(y)), 0), h - 1)
        # Index the array to get the distance value
        return field[yi, xi]

    def plan(self):
        """
        Run APF to compute an optimised path.

        Returns a dictionary with the path (a list of [x,y] points along the route), whether it successfully reached the goal and the number of iterations/steps it took
        """
        dist_field, grad_x, grad_y = self._precompute_fields()

        # Initial setup, current position is the start, goal is the goal. Converted to numpy array format as float type
        pos = np.asarray(self.start, dtype=float)
        goal = np.asarray(self.goal, dtype=float)
        path = [pos.copy()]

        # For each iteration we allow APF we check whether the goal has been reached within some tolerance.
        for iteration in range(self.config.apf_max_iter):
            # Check if we reached goal
            if self._arrived(pos):
                return {
                    "path": path,
                    "success": True,
                    "iters": iteration,
                    "switches": 0,
                }

            # Work out the resultant force and direction
            resultant_vector = self._apf_force(pos, goal, dist_field, grad_x, grad_y)
            resultant_mag = np.linalg.norm(resultant_vector)

            # if resultant force is negligible, we are stuck
            if resultant_mag < 1e-9:
                break

            # Work out the normalised step direction from the resultant vector
            step_dir = resultant_vector / resultant_mag

            # The new position is current position + step direction multiplied by the configured step distance
            new_pos = pos + step_dir * self.config.apf_step

            # change the position to the new position
            pos = new_pos
            # Append copy to the path so that the mutable reference isn't passed but a new copy is passed
            path.append(pos.copy())

        # TODO: consider shape

        # If loop is exited it has failed to find the goal within configured iteration since the success condition returns within the loop
        return {
            "path": path,
            "success": False,
            "iters": self.config.apf_max_iter,
            "switches": 0,
        }
