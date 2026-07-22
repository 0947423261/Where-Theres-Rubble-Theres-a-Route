"""
algorithms/hybrid.py

Hybrid APF + RRT* planner containing extensions

==================
The hybrid lets APF drive the vehicle normally and only invokes RRT* to escape APF local minima where it is stuck before reverting back to APF.

Two constructor parameters select between the four modes used in the ablation study:
    adaptive=False, bezier=False is the standard hybrid
    adaptive=True,  bezier=True is the modified hybrid (full method)
    adaptive=True,  bezier=False is the adaptive-only ablation
    adaptive=False, bezier=True is the bezier-only ablation

The extensions are adaptive stuck-detection and smoothing. RRT* escape aims for a sub-goal along the path to the goal because this is faster to compute and more efficient whilst having the same effect as aiming for the goal since we won't follow the complete RRT* path but rather switch back to APF.
"""

import numpy as np
from collections import deque

from .base_planner import BasePlanner
from .apf_mixin import APFMixin
from .rrt_star import RRTStar
from .smoothers import NoSmoother


class Hybrid(BasePlanner, APFMixin):
    # Initialise an object of the BasePlanner class to inherit from with the two mentioned parameters that enable/disable adaptive stuck-detection and smoothing
    def __init__(self, grid, start, goal, config, rng, adaptive=False, smoother=None):
        super().__init__(grid, start, goal, config, rng)

        # Set the flags for adaptive stuck-detection and smoothing
        self.adaptive = adaptive
        self.smoother = smoother if smoother is not None else NoSmoother(config)

    def _local_density(self, x, y, radius):
        """
        Fraction of cells that are obstacles within the radius parameter of (x, y), measured as a circle around the point.

        Cells off the edge of the map are counted as obstacles, so the planner treats the region next to a boundary wall as crowded.
        """

        # Grid dimensions for the bounds test
        h, w = self.grid.shape

        # Round the point and radius to integers
        center_x, center_y = int(round(x)), int(round(y))
        radius = int(round(radius))

        # Count for sampled cells within radius of the point and number of blocked cells
        total = 0
        blocked = 0

        # Iterate over x and y across the circle
        for change_y in range(-radius, radius + 1):
            for change_x in range(-radius, radius + 1):
                # The 2D array will go over a square that is diameter by diameter so we need to check for cells outside the circle
                # Keep only cells within radius of circle (circle equation)
                if change_x * change_x + change_y * change_y > radius * radius:
                    continue

                # We iterate over the cells within the circle
                check_x, check_y = center_x + change_x, center_y + change_y
                total += 1

                # if either the cell is outside bounds or the cell contains an obstacle we count it as blocked
                if (
                    not (0 <= check_x < w and 0 <= check_y < h)
                    or self.grid[check_y, check_x] != 0
                ):
                    blocked += 1

        # We can't have a circle with no cells
        if total == 0:
            return 0.0

        # Returns the fraction of blocked cells within the radius
        return blocked / total

    def _adaptive_window(self, density):
        """
        Feed local obstacle density (0..1) into a stuck detection function using linear interpolation between the configured bounds.

        N(rho) = round( N_min + (N_max - N_min) * (1 - rho) )

        Where:
            - N(rho) is the number of consecutive iterations with no progress before we consider the vehicle stuck.
            - N_min is the minimum threshold
            - N_max is the maximum threshold
            - rho is the value calculated as the local density


        A density near 0 (open space) gives N(rho) near N_max, so the planner avoids false alarms where progress is slow.

        A density near 1 (crowded) gives N near N_min, so the planner is sensitive and escapes a trap without spending much time
        """

        # Renaming for readability
        n_min = self.config.stuck_window_min
        n_max = self.config.stuck_window_max

        # Linear interpolation explained above
        window = n_min + (n_max - n_min) * (1.0 - density)

        # The window indexes a deque so it must be an integer
        return int(round(window))

    def _find_free_cell(self, target):
        """
        Return the nearest free (non-obstacle) cell to 'target', searching outward in square rings. This is so that RRT* has a valid sub-goal
        """

        # Gets the shape of the grid for the ring's boundaries
        h, w = self.grid.shape

        # Round the float target to integer
        target_x, target_y = int(round(target[0])), int(round(target[1]))

        # If the target cell is free (no obstacles and not out of bounds) then we can use it
        if not self._point_blocked((target_x, target_y)):
            return np.array([target_x, target_y], dtype=float)

        # Otherwise expand a square ring of increasing radius around the target
        for radius in range(1, max(h, w)):
            for change_x in range(-radius, radius + 1):
                for change_y in range(-radius, radius + 1):
                    # Candidate cell somewhere within the square ring around the target
                    nx, ny = target_x + change_x, target_y + change_y

                    # Accept the first candidate that is free (no obstacle and within bounds)
                    if not self._point_blocked((nx, ny)):
                        return np.array([nx, ny], dtype=float)

        # If every cell is an obstacle it will still return the target which will be caught later and result unsuccessful path find
        return np.asarray(target, dtype=float)

    def _pick_subgoal(self, pos):
        """
        Choose the point the RRT* escape should use as a goal, it is placed a fixed distance along a straight line from the current position and the global goal. This ensures progress instead of just local minima escape. If the goal is within a certain distance from the vehicle, we simply use the goal.
        """
        # Converts the current position to numpy array format as float type
        pos = np.asarray(pos, dtype=float)

        # Direction vector from the current position to the global goal
        direction = self.goal - pos

        # Magnitude (Euclidean distance) of that vector
        distance = np.linalg.norm(direction)

        # If the goal is nearer than the configured sub-goal distance, aim at it
        if distance <= self.config.subgoal_dist:
            return self._find_free_cell(self.goal)

        # Otherwise step the configured distance along the normalised direction after normalising the direction vector to a unit vector
        target = pos + (direction / distance) * self.config.subgoal_dist

        # Snap the target onto a free cell so RRT* has a reachable goal and isn't aiming for an obstacle or out of bounds
        return self._find_free_cell(target)

    def _escape(self, pos):
        """
        Run RRT* as a local escape from 'pos'. Tries aiming for a sub-goal first, and if
        that fails falls back to planning straight to the global goal, which
        completes the whole remaining route (less efficient)

        Returns the RRT* result dictionary (its 'success' key reports whether
        either attempt worked).
        """
        # Where the escape should aim for
        subgoal = self._pick_subgoal(pos)

        # A fresh RRT* algorithm object is constructed for each escape since its tree is rooted atthe current position. The same rng is passed so the whole run stays reproducible from a single seed.
        escape = RRTStar(self.grid, pos, subgoal, self.config, self.rng).plan()

        # Fallback: if the sub-goal was unreachable, try the real goal directly
        if not escape["success"]:
            escape = RRTStar(self.grid, pos, self.goal, self.config, self.rng).plan()

        # return the escape path
        return escape

    def plan(self):
        """
        Run the hybrid planner.

        Returns a dictionary with the path of points, whether it succeeded, the number of iterations, number of times it switched to RRT* and the junction points where the mode switch happened.
        """
        # Build the per-obstacle distance and gradient stacks once, so that every APF step is only an array lookup. This function is provided by mixin
        dist_stack, gx_stack, gy_stack = self._precompute_fields()

        ### === Initialisation ===
        # Current position is converted to numpy array of type float, the path contains the original start position
        pos = np.asarray(self.start, dtype=float)
        path = [pos.copy()]

        # A short memory of the most recent positions used for hysteresis, used to detect a lack of progress. It is capped at the largest window the adaptive rule can ask for so that the oldest entry is always available
        recent = deque(maxlen=self.config.stuck_window_max)
        # Add the starting position to the memory window
        recent.append(pos.copy())

        # coordinates in the path where mode switch occurred, used both for smoothing and annotation
        junction_indices = []

        # Counters for the number of switches and total number of iterations
        num_switches = 0
        total_iterations = 0
        ### === END ===

        # Every hybrid iteration is one APF step, so the APF budget bounds the run
        while total_iterations < self.config.apf_max_iter:
            # Increment the number of iterations
            total_iterations += 1

            # Arrival check + smoothing
            if self._arrived(pos):
                # If APF has arrived at the goal threshold attempt smoothing, if no smoothing is done the path is left untouched
                result_path = self.smoother.smooth(path, self.grid)

                return {
                    "path": result_path,
                    "success": True,
                    "iters": total_iterations,
                    "switches": num_switches,
                    "junctions": junction_indices,
                }

            ### === One APF step === ###
            # Work out the resultant force and its magnitude
            resultant_vector = self._apf_force(
                pos, self.goal, dist_stack, gx_stack, gy_stack
            )
            resultant_mag = np.linalg.norm(resultant_vector)

            # Flag to see if APF is stuck
            blocked = False

            # If the resultant force is negligible we are in a local minimum (equilibrium point) since no force is acting to move the vehicle
            if resultant_mag < 1e-9:
                blocked = True
            else:
                # Work out the unit step direction from the resultant force vector
                step_dir = resultant_vector / resultant_mag

                # The new position is current position + step direction multiplied by configured step distance
                new_pos = pos + step_dir * self.config.apf_step

                # If the step drives into an obstacle or out of bounds do not take it, consider APF blocked
                if self._point_blocked(new_pos):
                    blocked = True
                else:
                    # Otherwise commit to the step
                    pos = new_pos
                    # Append copies to the path and recent hysteresis window so the mutable reference isn't shared
                    path.append(pos.copy())
                    recent.append(pos.copy())

            # If there is adaptive stuck-detection
            if self.adaptive:
                # Measure the density of the surrounding environment
                density = self._local_density(
                    pos[0], pos[1], self.config.density_radius
                )
                # Set the window size to the size based on the adaptive window function
                window = self._adaptive_window(density)
            else:
                # Standard non-adaptive hybrid uses a fixed window size that is preconfigured
                window = self.config.stuck_window_fixed

            # Flag for stuck detection
            stuck = False

            # A hard block from moving into an obstacle or out of bounds or equilibrium resultant force is counted as stuck as there is no reason to wait
            if blocked:
                stuck = True
            # Otherwise only judge the movement as stuck (no-progress) if the window is full of coordinates that don't seem to make progress
            elif len(recent) >= window:
                # Finds the euclidean (straight-line) distance from the point that is a "window" steps ago and the current point. If there is oscillation a lot of distance will be covered with little displacement
                # moved is the displacement magnitude
                moved = np.linalg.norm(recent[-1] - recent[-window])

                # Below the configured threshold means no meaningful progress
                if moved < self.config.stuck_delta:
                    stuck = True

            # If it is stuck it switches to RRT* to escape and then back to APF
            if stuck:
                # Safety in case the path still enters a loop with RRT* escapes and is in a loop of switching modes
                if num_switches >= self.config.max_escapes:
                    break

                # Finds the escape path
                escape = self._escape(pos)

                # If the escape path from RRT* is not successful the hybrid has failed
                if not escape["success"]:
                    break

                # Increment the number of switches if we succeeded
                num_switches += 1

                # Record the coordinate (current point) where the mode switch occurs
                junction_indices.append(len(path) - 1)

                # Append the escape path, skipping its first point since that is the current position already in the path
                for point in escape["path"][1:]:
                    path.append(np.asarray(point, dtype=float))

                # Hand control back to APF from the end of the escape
                pos = np.asarray(path[-1], dtype=float)

                # The old positions in the window describe the trap we just left, so they would immediately re-trigger the stuck test. They are cleared
                recent.clear()
                # New position is added to the window
                recent.append(pos.copy())

            # If the loop is exited it has failed to reach the goal within the configured number of iterations, since the success condition returns from within the loop
        return {
            "path": path,
            "success": False,
            "iters": total_iterations,
            "switches": num_switches,
            "junctions": junction_indices,
        }
