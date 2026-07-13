"""
algorithms/apf_mixin.py

==========
This mixin class contains functionality used by both hybrid and standard apf algorithms
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


class APFMixin:
    def _precompute_fields(self):
        """
        Build the distance field and its gradient once, so every APF step is cheap.
        Returns (dist_field, grad_x, grad_y).

        dist_field is a 2D arrays of the euclidean distances to nearest obstacle
        grad_x and grad_y are 2D arrays holding the respective cartesian component of the gradient vector at that cell (essentially the repulsive force arrow for that cell when combined)
        """
        # grid == 0 means that free space is set to True and obstacles are False. The Scipy function finds the euclidean distance from every False cell.
        # Obstacles have 0 value and cells further away are "higher" and have large value
        dist_field = distance_transform_edt(self.grid == 0)
        # This calculates the slope at each cell telling you the direction the distance increases value increases the fastest. This is the direction of the repulsive force
        grad_y, grad_x = np.gradient(dist_field)
        return dist_field, grad_x, grad_y

    def _apf_force(pos, goal, dist_field, grad_x, grad_x, config):
        """
        Computes the resultant vector acting on the vehicle.
        """
        # Converts the current position and goal position into numpy arrays in float type
        pos = np.asarray(pos, dtype=float)
        goal = np.asarray(goal, dtype=float)

        # TODO: complete this function
        pass
