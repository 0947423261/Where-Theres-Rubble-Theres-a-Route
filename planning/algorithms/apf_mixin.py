"""
algorithms/apf_mixin.py

This mixin class contains functionality used by both hybrid and standard apf algorithms
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, label


class APFMixin:
    def _precompute_fields(self, max_obstacles=200):
        """
        Build the distance field and its gradient once, so every APF step is cheap.
        Returns (dist_field, grad_x, grad_y).

        dist_field is a 2D arrays of the euclidean distances to nearest obstacle
        grad_x and grad_y are 2D arrays holding the respective cartesian component of the gradient vector at that cell (essentially the repulsive force arrow for that cell when combined)
        """

        # Makes a mask on grid/map where obstacles are set to True and non-obstacles are set to False
        obstacle_mask = self.grid != 0

        # Labels connected groups of obstacle cells, each blob gets its own integer ID. Labels stores the 2D array map but labelled blob version
        labels, num_obstacles = label(obstacle_mask)

        # If there are no obstacles
        if num_obstacles == 0:
            # Makes a 3D grid with no obstacles, the 3rd dimension is the obstacles
            h, w = self.grid.shape
            empty = np.zeros((0, h, w), dtype=float)
            # Returns three copies of empty meaning there are no obstacles and 0 gradient or distance (grad_x, grad_y and the distance field are empty)
            return empty, empty.copy(), empty.copy()

        if num_obstacles > max_obstacles:
            # If there are too many obstacles, use the nearest obstacle
            # Free space becomes False and obstacles become true by inserting the mask then compute the euclidean distance from every cell to the nearest obstacle
            dist = distance_transform_edt(~obstacle_mask)
            # compute the cartesian gradient components for each cell
            gy, gx = np.gradient(dist)
            # Returns a single layer stack since only "one" obstacle exists (nearest)
            return dist[None, ...], gx[None, ...], gy[None, ...]

        # Creates lists for the distance field and gradients for each obstacle
        dist_list, gx_list, gy_list = [], [], []

        # Iterates through obstacles and make 3d cube where each slice is 2d array of distance, x component of gradient or y component of gradient
        for obstacle_id in range(1, num_obstacles + 1):
            # Create a boolean mask where it's true for cells that belong to this obstacle and false for anything else
            this_obstacle = labels == obstacle_id
            # Inverts the mask so that the obstacle cells are False and anything else is True. Then computes the euclidean distance from every cell to the nearest cell of the object
            dist = distance_transform_edt(~this_obstacle)

            # Create 2D arrays that hold gradient components for each cell going away from obstacles
            gy, gx = np.gradient(dist)

            # Append these values to the list for the obstacle
            dist_list.append(dist)
            gx_list.append(gx)
            gy_list.append(gy)

        return (np.stack(dist_list), np.stack(gx_list), np.stack(gy_list))

    def _sample_stack(self, stack, x, y):
        """r
        Read a value per obstacle from the stack at position (x, y).
        Returns an array of these values for each obstacle
        """
        # Finds then number of obstacles from the 1st dimension
        if stack.shape[0] == 0:
            # If no obstacles return an empty array
            return np.zeros(0)

        # Gets the height and width of the grid for boundary checks
        _, h, w = stack.shape
        # Gets the integer rounded x and y component of the bots position ensuring it is greater than 0 and not out of bounds
        xi = min(max(int(round(x)), 0), w - 1)
        yi = min(max(int(round(y)), 0), h - 1)

        # Returns the value (either distance or gradient component) for that cell
        return stack[:, yi, xi]

    def _apf_force(self, pos, goal, dist_stack, gx_stack, gy_stack):
        """
        Computes the resultant vector acting on the vehicle.
        """
        # Converts the current position and goal position into numpy arrays in float type
        pos = np.asarray(pos, dtype=float)
        goal = np.asarray(goal, dtype=float)

        # Calculate the attractive force proportional to how far the goal is. This follows Hook law style equation
        attractive_force = self.config.apf_k_att * (goal - pos)

        # Initialises the repulsive force vector
        repulsive_force = np.zeros(2)

        # Distances to each obstacle from current position to each obstacle as an array
        distances = self._sample_stack(dist_stack, pos[0], pos[1])
        # If the map has no obstacles nothing pushes the vehicle
        if distances.size == 0:
            return attractive_force

        # Variable renaming
        rho0 = self.config.apf_rho0

        in_range = []

        # Checks if the distance is within the configured threshold
        for distance in distances:
            if 0 < distance < rho0:
                in_range.append(True)
            else:
                in_range.append(False)

        in_range = np.array(in_range)

        # If there are any obstacles within range
        if np.any(in_range):
            # Extracts the distances that are in range
            rho = distances[in_range]

            # Khatib FIRAS function using vector of distances
            magnitudes = (
                self.config.apf_k_rep * (1.0 / rho - 1.0 / rho0) * (1.0 / (rho**2))
            )

            # Gets the x component of the gradient for each obstacle
            gx = self._sample_stack(gx_stack, pos[0], pos[1])[in_range]
            # Gets the y component of the gradient for each obstacle
            gy = self._sample_stack(gy_stack, pos[0], pos[1])[in_range]

            # Creates an array of magnitudes by finding the hypotenuse of each x,y gradient pair for each obstacle
            gradient_magnitudes = np.hypot(gx, gy)

            # Only uses magnitudes that are non-negligible
            usable = gradient_magnitudes > 1e-9
            # If there are such magnitudes
            if np.any(usable):
                # Find the unit direction vector for the gradient and multiply by the magnitude of the force found from FIRAS to get a list of components of force vectors for both the X and Y component
                fx = magnitudes[usable] * (gx[usable] / gradient_magnitudes[usable])
                fy = magnitudes[usable] * (gy[usable] / gradient_magnitudes[usable])
                # Sums the components of the resultant force to make the resultant force vector
                repulsive_force = np.array([fx.sum(), fy.sum()])

        return attractive_force + repulsive_force
