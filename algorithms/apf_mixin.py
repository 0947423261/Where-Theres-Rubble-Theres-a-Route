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

    def _apf_force(self, pos, goal, dist_field, grad_x, grad_y):
        """
        Computes the resultant vector acting on the vehicle.
        """
        # Converts the current position and goal position into numpy arrays in float type
        pos = np.asarray(pos, dtype=float)
        goal = np.asarray(goal, dtype=float)

        # Calculate the attractive force proportional to how far the goal is. This follows Hook law style equation
        attractive_force = self.config.apf_k_att * (goal - pos)

        # Repulsive force from objects
        dist_to_near_obj = self._sample(dist_field, pos[0], pos[1])

        # Initialises the repulsive force vector
        repulsive_force = np.zeros(2)

        # Checks if the distance is within the configured threshold
        if 0 < dist_to_near_obj < config.apf_rho0:
            # Khatib FIRAS function
            magnitude = (
                config.apf_k_rep
                * (1.0 / dist_to_near_obj - 1.0 / config.apf_rho0)
                * (1.0 / (dist_to_near_obj**2))
            )

            # Gets the x component of the gradient
            gx = self._sample(grad_x, pos[0], pos[1])
            # Gets the y component of the gradient
            gy = self._sample(grad_y, pos[0], pos[1])

            # Constructs gradient vector
            gradient_vector = np.array([gx, gy])

            # Gets the magnitude of the gradient vector
            gradient_magnitude = np.linalg.norm(gradient_vector)

            if gradient_magnitude > 1e-9:
                # Normalises the gradient vector and then multiplies by the magnitude calculated by Khatib (preserving repulsive force direction and using Khatib's exponential magnitude for APF)
                repulsive_force = magnitude * (gradient_vector / gradient_magnitude)

        return attractive_force + repulsive_force
