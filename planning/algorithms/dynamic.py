"""
algorithms/dynamic.py

Drivers for the moving obstacle scenario

The static planners plan once and stop. Here the vehicle has to be somewhere at every time step t while the obstacle moves, so the driver decides what to do at step t given where the obstacle is at step t.

Two drivers:
    ReactiveAPF     recomputes the APF force every step from the static field plus the obstacle's current position. Never replans
    PlanOnceDriver  plans a route over the static map with one of the other planners and drives it. Optionally replans before contact when the route ahead is threatened, on a fixed period, and after a hit

Both return the same dictionary:
    'path'      the route driven, one point per time step
    'success'   reached the goal and never hit
    'reached'   arrived at the goal
    'collided'  hit by the obstacle
    'iters'     time steps taken
    'replans'   number of replans (0 for ReactiveAPF)
    'plan_time' seconds spent in planning calls. For ReactiveAPF every step is a planning call so this is the whole run; for PlanOnceDriver it is the sum of the wrapped planner's calls, the driving loop excluded
    'switches'  RRT* escapes inside the wrapped planner, so the columns match the static runs
"""

from abc import abstractmethod

import time

import numpy as np

from .base_planner import BasePlanner
from .apf_mixin import APFMixin


class DynamicDriver(BasePlanner):
    """Abstract base for the dynamic drivers. Inherits the arrival and collision checks from BasePlanner and adds the obstacle"""

    def __init__(self, grid, start, goal, config, rng, obstacle):
        super().__init__(grid, start, goal, config, rng)

        # The grid stays static, the obstacle is queried separately per time step
        self.obstacle = obstacle

        # Seconds spent planning, added to by each driver as it goes
        self.plan_time = 0.0

    @abstractmethod
    def plan(self):
        """Drive from start to goal and return the result dictionary described at the top of the file"""
        raise NotImplementedError

    def _hit(self, pos, t):
        """True if the obstacle overlaps the vehicle at time step t"""
        return self.obstacle.hits(pos[0], pos[1], t, self.config.vehicle_radius)

    def _result(self, path, reached, collided, steps, replans=0, switches=0):
        """Pack the outcome into the dictionary the runner expects"""
        return {
            "path": path,
            "success": reached and not collided,
            "reached": reached,
            "collided": collided,
            "iters": steps,
            "replans": replans,
            "switches": switches,
            "plan_time": self.plan_time,
        }


class ReactiveAPF(DynamicDriver, APFMixin):
    """
    APF recomputed every time step. Static repulsion comes from the precomputed stacks in the mixin, the moving obstacle's repulsion is added fresh each step from its current position
    """

    def _result(self, path, reached, collided, steps, replans=0, switches=0):
        # Recomputing the force every step is the planning, so the run's whole time is its planning time
        self.plan_time = time.perf_counter() - self._started
        return super()._result(path, reached, collided, steps, replans, switches)

    def _moving_repulsion(self, pos, t):
        """
        Repulsive force from the moving obstacle at time step t.

        The obstacle is a square so the nearest point is the vehicle position clamped into the square. Same FIRAS function as the static field but with a wider cut-off and a stronger gain, since being hit ends the run
        """
        # Obstacle centre at this time step
        center_x, center_y = self.obstacle.position_at(t)

        # Closest point on the square to the vehicle
        nearest = np.array(
            [
                np.clip(pos[0], center_x - self.obstacle.half, center_x + self.obstacle.half),
                np.clip(pos[1], center_y - self.obstacle.half, center_y + self.obstacle.half),
            ]
        )

        # Vector from obstacle to vehicle, the direction the push acts in
        away = pos - nearest
        distance = np.linalg.norm(away)

        # Variable renaming for readability
        rho0 = self.config.dynamic_rho0

        # Outside the cut-off, or on the surface where the direction is undefined, no push
        if not 0 < distance < rho0:
            return np.zeros(2)

        # FIRAS function scaled up since a collision here is fatal
        magnitude = (
            self.config.apf_k_rep
            * self.config.dynamic_k_rep_scale
            * (1.0 / distance - 1.0 / rho0)
            * (1.0 / (distance**2))
        )

        # Magnitude along the unit direction away from the obstacle
        return magnitude * (away / distance)

    def plan(self):
        """
        Drive reactively. Each step recomputes the force from the static field and the obstacle's current position, no route is committed to
        """
        # Every step of a reactive run is planning, so the whole run is timed from here. _result reads the clock when it packs the outcome
        self._started = time.perf_counter()

        # Build the static distance and gradient stacks once, same as the APF planner
        dist_stack, gx_stack, gy_stack = self._precompute_fields()

        ### === Initialisation === ###
        # Current position is the start, the path records the position at every time step
        pos = np.asarray(self.start, dtype=float)
        path = [pos.copy()]
        ### === END ===

        for t in range(self.config.dynamic_max_steps):
            # Check if we reached goal
            if self._arrived(pos):
                return self._result(path, reached=True, collided=False, steps=t)

            # Static part of the resultant force from the precomputed stacks
            resultant_vector = self._apf_force(
                pos, self.goal, dist_stack, gx_stack, gy_stack
            )

            # Reactive part from the moving obstacle
            resultant_vector = resultant_vector + self._moving_repulsion(pos, t)
            resultant_mag = np.linalg.norm(resultant_vector)

            # if resultant force is negligible, we are stuck
            if resultant_mag < 1e-9:
                return self._result(path, reached=False, collided=False, steps=t)

            # Work out the normalised step direction and take one step
            step_dir = resultant_vector / resultant_mag
            new_pos = pos + step_dir * self.config.apf_step

            # If the step drives into an obstacle stop, there is no escape mode here. The whole segment is checked, same as the static planners
            if self._segment_blocked(pos, new_pos):
                return self._result(path, reached=False, collided=False, steps=t)

            # Take the step and record it
            pos = new_pos
            path.append(pos.copy())

            # Hit by the obstacle ends the run. The vehicle now stands where it is at time t + 1, so it is compared with the obstacle at t + 1 as well
            if self._hit(pos, t + 1):
                return self._result(path, reached=False, collided=True, steps=t + 1)

        # Ran out of time steps without arriving
        return self._result(
            path, reached=False, collided=False, steps=self.config.dynamic_max_steps
        )


class PlanOnceDriver(DynamicDriver):
    """
    Plan a route over the static map with one of the other planners and drive it.

    The planner never sees the moving obstacle since it will have moved by the time the vehicle arrives. replan=False drives the committed route regardless. replan=True plans again from the current position, at the cost of a full planning call each time: before contact when the route ahead runs within reach of the obstacle, on the fixed period if one is configured, and after a hit
    """

    def __init__(
        self, grid, start, goal, config, rng, obstacle, planner_class, replan=False, **planner_kwargs
    ):
        super().__init__(grid, start, goal, config, rng, obstacle)

        # Planner class to build routes with plus its extra constructor arguments (adaptive flag, smoother for the hybrid)
        self.planner_class = planner_class
        self.planner_kwargs = planner_kwargs

        # Whether a hit ends the run or triggers a replan
        self.replan = replan

    def _plan_route(self, grid, start):
        """Run the wrapped planner from 'start' over 'grid' and return its result dictionary. Construction and planning are timed together, the same as the static runs"""
        started = time.perf_counter()
        result = self.planner_class(
            grid, start, self.goal, self.config, self.rng, **self.planner_kwargs
        ).plan()
        self.plan_time += time.perf_counter() - started

        return result

    def _route_threatened(self, route, index, t, since_plan):
        """
        True when the driver should plan again before taking its next step.

        The fixed period fires on its own. Otherwise the next replan_horizon steps of the route are tested against the obstacle's current footprint grown by how far it can travel in that many steps, which is the set of places it could be while the vehicle drives that stretch. The obstacle's actual walk is never read ahead of time, since the vehicle cannot know it. A route planned less than replan_horizon steps ago already goes around where the obstacle stands, so it is left alone
        """
        period = self.config.replan_period
        if period > 0 and since_plan >= period:
            return True

        horizon = self.config.replan_horizon
        if since_plan < horizon:
            return False

        # How far the obstacle can get in the horizon, on top of the vehicle's own radius
        reach = self.config.vehicle_radius + self.obstacle.speed * horizon

        # The stretch of route the vehicle drives in the horizon, one route point per step at most
        ahead = 0.0
        last = route[min(index, len(route) - 1)]
        for point in route[index:]:
            ahead += np.linalg.norm(point - last)
            last = point
            if ahead > self.config.apf_step * horizon:
                break
            if self.obstacle.hits(point[0], point[1], t, reach):
                return True

        return False

    def _replan_grid(self, pos, t):
        """
        Map to replan over after a hit: the static map with the obstacle stamped in at its current position.

        The cells around the vehicle are cleared afterwards since the obstacle is on top of the vehicle at that moment and a planner cannot start from a blocked cell
        """
        # Stamp the obstacle into a copy of the static map
        grid = self.obstacle.stamp(self.grid, t, self.config.vehicle_radius)

        # Free the 3x3 block around the vehicle so the replan has a start
        h, w = grid.shape
        xi, yi = int(round(pos[0])), int(round(pos[1]))
        x0, x1 = max(0, xi - 1), min(w, xi + 2)
        y0, y1 = max(0, yi - 1), min(h, yi + 2)
        grid[y0:y1, x0:x1] = 0

        return grid

    def plan(self):
        """
        Plan once over the static map then drive the route one step per time step while the obstacle moves
        """
        ### === Initialisation === ###
        # First route, planned against the map at time zero
        result = self._plan_route(self.grid, self.start)

        # No route means nothing to drive
        if not result["success"] or result["path"] is None:
            return self._result(
                result["path"], reached=False, collided=False, steps=0,
                switches=result.get("switches", 0),
            )

        # Route being followed as float numpy arrays
        route = [np.asarray(point, dtype=float) for point in result["path"]]

        # Index of the route point being driven towards
        index = 0

        # Current position, the path driven and the counters
        pos = np.asarray(self.start, dtype=float)
        driven = [pos.copy()]
        replans = 0
        switches = result.get("switches", 0)

        # Set once the obstacle touches the vehicle. A replan driver carries on after contact so reached and replans are still measured, but the run has already failed
        collided = False

        # Steps since the route was last planned, read by the threat check and the fixed period
        since_plan = 0
        ### === END ===

        for t in range(self.config.dynamic_max_steps):
            # Check if we reached goal
            if self._arrived(pos):
                return self._result(
                    driven, reached=True, collided=collided, steps=t,
                    replans=replans, switches=switches,
                )

            # With replanning on, plan again before moving when the route ahead is threatened, rather than waiting to be hit
            if self.replan and self._route_threatened(route, index, t, since_plan):
                replans += 1
                since_plan = 0
                result = self._plan_route(self._replan_grid(pos, t), pos)

                # A replan that finds nothing keeps the old route, since nothing has happened yet
                if result["success"] and result["path"] is not None:
                    route = [np.asarray(point, dtype=float) for point in result["path"]]
                    index = 0
                    switches += result.get("switches", 0)

            # Skip past route points already reached so the vehicle does not stall on closely spaced points
            target = route[min(index, len(route) - 1)]
            while (
                index < len(route) - 1
                and np.linalg.norm(target - pos) < self.config.apf_step
            ):
                index += 1
                target = route[index]

            # Step towards the target by apf_step so every driver moves at the same speed
            step_vector = target - pos
            step_mag = np.linalg.norm(step_vector)
            if step_mag > 1e-9:
                pos = pos + (step_vector / step_mag) * self.config.apf_step

            driven.append(pos.copy())
            since_plan += 1

            # Nothing more to do unless the obstacle hit the vehicle. Both have moved on to time t + 1 by now
            if not self._hit(pos, t + 1):
                continue

            # Any contact is a failure whatever happens next
            collided = True

            # Without replanning a hit ends the run
            if not self.replan:
                return self._result(
                    driven, reached=False, collided=True, steps=t + 1,
                    replans=replans, switches=switches,
                )

            # Otherwise replan from here over the map with the obstacle stamped in
            replans += 1
            since_plan = 0
            result = self._plan_route(self._replan_grid(pos, t + 1), pos)

            # Failed replan leaves the vehicle stranded under the obstacle
            if not result["success"] or result["path"] is None:
                return self._result(
                    driven, reached=False, collided=True, steps=t + 1,
                    replans=replans, switches=switches,
                )

            # Follow the new route from the start
            route = [np.asarray(point, dtype=float) for point in result["path"]]
            index = 0
            switches += result.get("switches", 0)

        # Ran out of time steps without arriving
        return self._result(
            driven, reached=False, collided=collided,
            steps=self.config.dynamic_max_steps, replans=replans, switches=switches,
        )
