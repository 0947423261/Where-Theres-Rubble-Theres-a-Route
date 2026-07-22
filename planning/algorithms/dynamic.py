"""
algorithms/dynamic.py

Driving the vehicle while an obstacle moves

============
For static maps the route can be precalculated but in disasters the map can change during the traversal, here the route must be adapted.

ReactiveAPF recomputes the resultant force every step from the static field instead of caching, so it steers around the obstacle without ever replanning.

PlanOnceDriver plans a full route over the static map with any of the other planners and then drives it. The route was computed against a snapshot, so when the obstacle moves onto the route, the vehicle has an accident.
"""

from abc import abstractmethod

import numpy as np

from .base_planner import BasePlanner
from .apf_mixin import APFMixin


class DynamicDriver(BasePlanner):
    """Abstract base for driving under a moving obstacle. It is a BasePlanner that also knows about the obstacle, so it inherits the arrival and collision checks."""

    def __init__(self, grid, start, goal, config, rng, obstacle):
        super().__init__(grid, start, goal, config, rng)

        # The obstacle that moves. The grid held by the base class stays static and the obstacle is asked about separately
        self.obstacle = obstacle

    @abstractmethod
    def plan(self):
        """Drive from the start to the goal and return the result dictionary described in the module docstring."""
        raise NotImplementedError

    def _hit(self, pos, t):
        """Return True if the obstacle is on top of the vehicle at time step t."""
        return self.obstacle.hits(pos[0], pos[1], t, self.config.vehicle_radius)

    def _result(self, path, reached, collided, steps, replans=0, switches=0):
        """Bundle the outcome of a run into the dictionary the runner expects."""
        return {
            "path": path,
            "success": reached and not collided,
            "reached": reached,
            "collided": collided,
            "iters": steps,
            "replans": replans,
            "switches": switches,
        }


class ReactiveAPF(DynamicDriver, APFMixin):
    """
    APF recomputed every time step. The static repulsion comes from the precomputed stacks the mixin builds once, and the repulsion from the moving obstacle is added fresh at every step from wherever it is now.
    """

    def _moving_repulsion(self, pos, t):
        """
        The push away from the moving obstacle at time step t.

        The obstacle is a square, so the nearest point of it is found by clamping the vehicle position into the square. The magnitude is the same Khatib FIRAS function the static field uses, with its own wider cut-off distance and a stronger gain, because a static obstacle that is brushed only costs clearance while the moving one ends the run.
        """
        # Where the obstacle is at this time step
        center_x, center_y = self.obstacle.position_at(t)

        # The point on the square closest to the vehicle
        nearest = np.array(
            [
                np.clip(
                    pos[0], center_x - self.obstacle.half, center_x + self.obstacle.half
                ),
                np.clip(
                    pos[1], center_y - self.obstacle.half, center_y + self.obstacle.half
                ),
            ]
        )

        # The vector pointing from the obstacle to the vehicle, which is the direction the push acts in
        away = pos - nearest
        distance = np.linalg.norm(away)

        # Variable renaming for readability
        rho0 = self.config.dynamic_rho0

        # Outside the cut-off, or exactly on the surface where the direction is undefined, nothing pushes
        if not 0 < distance < rho0:
            return np.zeros(2)

        # Khatib FIRAS function, scaled up because a collision here is fatal to the run
        magnitude = (
            self.config.apf_k_rep
            * self.config.dynamic_k_rep_scale
            * (1.0 / distance - 1.0 / rho0)
            * (1.0 / (distance**2))
        )

        # Turn the magnitude into a vector along the unit direction away from the obstacle
        return magnitude * (away / distance)

    def plan(self):
        """
        Drive reactively. Every step recomputes the force from the static field and the obstacle's current position, so no route is ever committed to.
        """
        # Build the static distance and gradient stacks once, exactly as the static APF planner does
        dist_stack, gx_stack, gy_stack = self._precompute_fields()

        ### === Initialisation === ###
        # Current position starts at the start, and the path records where the vehicle was at every time step
        pos = np.asarray(self.start, dtype=float)
        path = [pos.copy()]
        ### === END ===

        for t in range(self.config.dynamic_max_steps):
            # Arrival check, which is the only successful way out of the loop
            if self._arrived(pos):
                return self._result(path, reached=True, collided=False, steps=t)

            # The static part of the resultant force, computed by the mixin from the precomputed stacks
            resultant_vector = self._apf_force(
                pos, self.goal, dist_stack, gx_stack, gy_stack
            )

            # The reactive part, which is the whole point of this driver
            resultant_vector = resultant_vector + self._moving_repulsion(pos, t)
            resultant_mag = np.linalg.norm(resultant_vector)

            # A negligible resultant force means the vehicle is in an equilibrium point and nothing is going to move it
            if resultant_mag < 1e-9:
                return self._result(path, reached=False, collided=False, steps=t)

            # Work out the unit step direction and take one step along it
            step_dir = resultant_vector / resultant_mag
            new_pos = pos + step_dir * self.config.apf_step

            # A step into a wall is not taken, and there is no escape mode here to recover with
            if self._point_blocked(new_pos):
                return self._result(path, reached=False, collided=False, steps=t)

            # Commit to the step and record it
            pos = new_pos
            path.append(pos.copy())

            # Being caught by the obstacle ends the run
            if self._hit(pos, t):
                return self._result(path, reached=False, collided=True, steps=t)

        # The loop ran out of time steps without arriving
        return self._result(
            path, reached=False, collided=False, steps=self.config.dynamic_max_steps
        )


class PlanOnceDriver(DynamicDriver):
    """
    Plan a route over the static map with any of the other planners and then drive it.

    The planner sees the map without the moving obstacle in it, because a global planner cannot plan around something that will have moved by the time the vehicle arrives. With replan set to False the vehicle drives its committed route and takes whatever is coming. With replan set to True it plans again from where it stands each time it is caught, which recovers the run at the cost of a full planning call.
    """

    def __init__(
        self,
        grid,
        start,
        goal,
        config,
        rng,
        obstacle,
        planner_class,
        replan=False,
        **planner_kwargs,
    ):
        super().__init__(grid, start, goal, config, rng, obstacle)

        # The planner class to build routes with, along with any extra constructor arguments such as the smoother or the adaptive flag for the hybrid
        self.planner_class = planner_class
        self.planner_kwargs = planner_kwargs

        # Whether being hit ends the run or triggers a fresh plan
        self.replan = replan

    def _plan_route(self, grid, start):
        """Run the wrapped planner from 'start' over 'grid' and return its result dictionary."""
        return self.planner_class(
            grid, start, self.goal, self.config, self.rng, **self.planner_kwargs
        ).plan()

    def _replan_grid(self, pos, t):
        """
        The map to replan over after a collision: the static map with the obstacle stamped in where it is now.

        The cells immediately around the vehicle are cleared again afterwards, because the obstacle is sitting on top of the vehicle at the moment of the collision and a planner cannot grow a tree from a blocked root.
        """
        # Stamp the obstacle into a copy of the static map
        grid = self.obstacle.stamp(self.grid, t)

        # Free the cells the vehicle occupies so the replan has somewhere to start from
        h, w = grid.shape
        xi, yi = int(round(pos[0])), int(round(pos[1]))
        x0, x1 = max(0, xi - 1), min(w, xi + 2)
        y0, y1 = max(0, yi - 1), min(h, yi + 2)
        grid[y0:y1, x0:x1] = 0

        return grid

    def plan(self):
        """
        Plan once over the static map and then drive the route one step per time step while the obstacle moves.
        """
        ### === Initialisation === ###
        # The first route, planned against the map as it looks at time zero
        result = self._plan_route(self.grid, self.start)

        # A planner that could not find a route at all never gets to drive
        if not result["success"] or result["path"] is None:
            return self._result(
                result["path"],
                reached=False,
                collided=False,
                steps=0,
                switches=result.get("switches", 0),
            )

        # The route being followed, as numpy arrays of type float
        route = [np.asarray(point, dtype=float) for point in result["path"]]

        # Which point of the route the vehicle is currently driving towards
        index = 0

        # Current position, the path actually driven, and the counters
        pos = np.asarray(self.start, dtype=float)
        driven = [pos.copy()]
        replans = 0
        switches = result.get("switches", 0)
        ### === END ===

        for t in range(self.config.dynamic_max_steps):
            # Arrival check, which is the only successful way out of the loop
            if self._arrived(pos):
                return self._result(
                    driven,
                    reached=True,
                    collided=False,
                    steps=t,
                    replans=replans,
                    switches=switches,
                )

            # Walk the target forward past any route points the vehicle has already reached, so it does not stall on a cluster of closely spaced points
            target = route[min(index, len(route) - 1)]
            while (
                index < len(route) - 1
                and np.linalg.norm(target - pos) < self.config.apf_step
            ):
                index += 1
                target = route[index]

            # Step towards the target by the same step distance APF uses, so every driver moves at the same speed and the comparison is fair
            step_vector = target - pos
            step_mag = np.linalg.norm(step_vector)
            if step_mag > 1e-9:
                pos = pos + (step_vector / step_mag) * self.config.apf_step

            driven.append(pos.copy())

            # Nothing more to do this step unless the obstacle caught the vehicle
            if not self._hit(pos, t):
                continue

            # Without replanning, being hit ends the run
            if not self.replan:
                return self._result(
                    driven,
                    reached=False,
                    collided=True,
                    steps=t,
                    replans=replans,
                    switches=switches,
                )

            # Otherwise plan a fresh route from here, over the map with the obstacle where it is now
            replans += 1
            result = self._plan_route(self._replan_grid(pos, t), pos)

            # A replan that fails leaves the vehicle stranded under the obstacle
            if not result["success"] or result["path"] is None:
                return self._result(
                    driven,
                    reached=False,
                    collided=True,
                    steps=t,
                    replans=replans,
                    switches=switches,
                )

            # Follow the new route from its first point
            route = [np.asarray(point, dtype=float) for point in result["path"]]
            index = 0
            switches += result.get("switches", 0)

        # The loop ran out of time steps without arriving
        return self._result(
            driven,
            reached=False,
            collided=False,
            steps=self.config.dynamic_max_steps,
            replans=replans,
            switches=switches,
        )
