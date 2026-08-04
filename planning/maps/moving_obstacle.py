"""
maps/moving_obstacle.py

A single obstacle that moves over time, plus the scenario it patrols

The static scenarios never show what APF is actually good at. APF recomputes its move from the field it can see at that instant, so an obstacle that moves is simply a field that changed and the vehicle steers around it without being asked. RRT* and the hybrid plan a whole route over one snapshot of the world, so when the world moves underneath them the route they committed to can put the vehicle straight under the obstacle. That contrast is the reason this scenario exists.

The obstacle wanders on a seeded random walk and bounces off the edges of the box it is confined to, so its motion is unpredictable to the planner while still being identical on every run of the same seed.

The obstacle is stamped into the grid with cell value 3. The collision helpers treat every non-zero cell as blocked so it blocks like any other obstacle, and the separate value only exists so the animation can colour it differently.
"""

import numpy as np

from .inflate import inflate_obstacles

# Cell value written into the grid where the moving obstacle currently sits
MOVING_VALUE = 3


class MovingObstacle:
    """A square obstacle that walks around inside a box, remembering where it was at every time step so the motion can be replayed."""

    def __init__(self, grid_shape, seed, half_size, speed, start, bounds=None):
        """
        grid_shape : the (height, width) of the map it moves in
        seed       : makes the walk reproducible
        half_size  : the obstacle is a square of (2 * half_size + 1) cells
        speed      : how many cells it travels per time step
        start      : the (x, y) centre it starts from
        bounds     : optional (x_min, x_max, y_min, y_max) box the centre stays inside, used to make it patrol one stretch of a corridor instead of the whole map
        """
        # Map dimensions, used to keep the obstacle from walking off the edge
        self.h, self.w = grid_shape

        # Half the side length of the square, and the distance travelled each step
        self.half = half_size
        self.speed = speed

        # Its own generator, so the obstacle's motion is independent of whatever the planner is sampling
        self.rng = np.random.default_rng(seed)

        # The optional box the centre is confined to
        self.bounds = bounds

        # The starting centre as a numpy array of type float
        self.pos = np.asarray(start, dtype=float)

        # An initial heading drawn at random, turned into a velocity vector of the configured speed
        angle = self.rng.uniform(0, 2 * np.pi)
        self.vel = np.array([np.cos(angle), np.sin(angle)]) * speed

        # Time step to centre position, so a position that has been computed once is never recomputed and never changes
        self._trail = {0: self.pos.copy()}

    def _limits(self):
        """Return the (x_min, x_max, y_min, y_max) the centre is allowed to move between."""
        # Keep the whole square inside the map with a two cell margin
        margin = self.half + 2

        # Without a box the obstacle may use the whole map
        if self.bounds is None:
            return margin, self.w - margin, margin, self.h - margin

        # With a box, take whichever of the two limits is tighter on each side
        x_min, x_max, y_min, y_max = self.bounds
        return (
            max(margin, x_min),
            min(self.w - margin, x_max),
            max(margin, y_min),
            min(self.h - margin, y_max),
        )

    def position_at(self, t):
        """
        Return the centre (x, y) of the obstacle at integer time step t.

        The walk is generated one step at a time and every step is kept, so asking for the same time twice always gives the same answer and asking for a time already reached costs nothing.
        """
        # A step that has already been walked is just a lookup
        if t in self._trail:
            return self._trail[t]

        # Otherwise walk forward from the last step that was computed
        last = max(self._trail)
        pos = self._trail[last].copy()
        vel = self.vel.copy()

        # The box the centre has to stay inside
        x_min, x_max, y_min, y_max = self._limits()

        for step in range(last + 1, t + 1):
            # A small random turn each step, which is what makes the walk unpredictable rather than a straight line
            turn = self.rng.uniform(-0.5, 0.5)
            cos_turn, sin_turn = np.cos(turn), np.sin(turn)

            # Rotate the velocity vector by that turn using the 2D rotation matrix
            vel = np.array(
                [
                    cos_turn * vel[0] - sin_turn * vel[1],
                    sin_turn * vel[0] + cos_turn * vel[1],
                ]
            )

            # Renormalise so the turn changes the heading and never the speed
            vel = vel / (np.linalg.norm(vel) + 1e-9) * self.speed

            # Move the centre by one step of the velocity
            pos = pos + vel

            # Bounce off the left and right limits by reflecting the x component and pinning the centre back inside
            if pos[0] < x_min or pos[0] > x_max:
                vel[0] = -vel[0]
                pos[0] = np.clip(pos[0], x_min, x_max)

            # Same on the top and bottom limits
            if pos[1] < y_min or pos[1] > y_max:
                vel[1] = -vel[1]
                pos[1] = np.clip(pos[1], y_min, y_max)

            # Remember where the obstacle was at this step
            self._trail[step] = pos.copy()

        # Carry the velocity forward so the next call continues the same walk
        self.vel = vel

        return self._trail[t]

    def stamp(self, base_grid, t, vehicle_radius=0.0):
        """
        Return a copy of base_grid with the obstacle painted in at its time t position. The grid passed in is left untouched, so the static map can be reused for every step.

        A cell is painted when a vehicle of the given radius standing on it would overlap the obstacle, which is the same question hits() answers. Using the one test for both means a route that clears the stamped grid also clears the obstacle when it is driven.
        """
        # Work on a copy so the caller keeps a clean static map
        grid = base_grid.copy()

        # Where the centre is at this time step
        center_x, center_y = self.position_at(t)

        # The range of cells that could possibly overlap, clipped to the map. The reach is the half size plus the vehicle radius, rounded outwards
        reach = int(np.ceil(self.half + vehicle_radius))
        x0, x1 = max(0, int(np.floor(center_x)) - reach), min(self.w, int(np.ceil(center_x)) + reach + 1)
        y0, y1 = max(0, int(np.floor(center_y)) - reach), min(self.h, int(np.ceil(center_y)) + reach + 1)

        for y in range(y0, y1):
            for x in range(x0, x1):
                if self.hits(x, y, t, vehicle_radius):
                    grid[y, x] = MOVING_VALUE

        return grid

    def hits(self, x, y, t, vehicle_radius):
        """
        Return True if a vehicle of the given radius centred on (x, y) overlaps the obstacle at time t.

        The test clamps the vehicle centre into the square to find the closest point of the obstacle, then compares that distance against the radius, which is the standard circle against axis-aligned box overlap test.
        """
        # Where the obstacle is at this time step
        center_x, center_y = self.position_at(t)

        # The point of the square closest to the vehicle
        nearest_x = np.clip(x, center_x - self.half, center_x + self.half)
        nearest_y = np.clip(y, center_y - self.half, center_y + self.half)

        # They overlap once that point is within one vehicle radius
        return np.hypot(x - nearest_x, y - nearest_y) <= vehicle_radius


class DynamicCorridor:
    """
    The dynamic scenario: one wide horizontal corridor from the start to the goal with an obstacle patrolling across the middle of it.

    The map is kept deliberately open. A dense map would tangle the moving obstacle up with APF's local minima problem and it would no longer be clear which of the two caused a failure. With one obvious route, a planner that commits to a route meets the obstacle head on and a planner that reacts can weave around it.
    """

    # Constructor just holds the configuration
    def __init__(self, config):
        self.config = config

    def build(self, seed, speed_factor=1.0):
        """
        Build the corridor and its obstacle from the seed. Returns (grid, start, goal, obstacle). The obstacle moves at speed_factor times the vehicle's speed, which is apf_step cells per time step.
        """
        # The map is square, like the static scenarios
        n = self.config.grid_size

        # An empty map to carve the corridor out of, and a generator for the pillars
        grid = np.zeros((n, n), dtype=np.uint8)
        rng = np.random.default_rng(seed)

        # Solid walls across the top and the bottom, which leaves a clear horizontal channel down the middle. The channel is deliberately narrow: in a wide one the vehicle and the obstacle would pass each other most of the time and the scenario would measure luck rather than behaviour
        wall = max(4, n // 2 - self.config.dynamic_corridor_half)
        grid[0:wall, :] = 1
        grid[n - wall : n, :] = 1

        # One small static pillar in the channel, well away from the stretch the obstacle patrols so the two can never seal the corridor between them
        pillar_x = int(rng.integers(n * 0.12, n * 0.24))
        pillar_y = int(rng.integers(wall + 1, n - wall - 4))
        grid[pillar_y : pillar_y + 3, pillar_x : pillar_x + 3] = 1

        # The route runs the length of the corridor, from just inside the left edge to just inside the right
        start = np.array([4, n // 2], dtype=float)
        goal = np.array([n - 5, n // 2], dtype=float)

        # Clear a small square at each end so a pillar can never land on top of them
        for end in (start, goal):
            xi, yi = int(end[0]), int(end[1])
            grid[max(0, yi - 3) : yi + 4, max(0, xi - 3) : xi + 4] = 0

        # Grow the walls and the pillar by the vehicle radius, same as the static maps. The moving obstacle is not on this grid; it carries its own radius test in hits()
        inflate_obstacles(grid, self.config.vehicle_radius)

        # The obstacle patrols up and down across the middle of the corridor. Its seed is offset from the map seed so the map and the motion are independent but both reproducible
        mid_x = n // 2
        obstacle = MovingObstacle(
            grid.shape,
            seed=seed + 777,
            half_size=self.config.dynamic_obstacle_half,
            speed=self.config.apf_step * speed_factor,
            start=(mid_x, n // 2),
            bounds=(mid_x - 6, mid_x + 6, wall + 3, n - wall - 3),
        )

        return grid, start, goal, obstacle


def make_dynamic_instance(seed, config, speed_factor=1.0):
    """
    Build the dynamic scenario from the seed and return (grid, start, goal, obstacle), with the obstacle at speed_factor times the vehicle's speed. This mirrors make_instance in maps/scenarios.py so the runners treat the two the same way.
    """
    return DynamicCorridor(config).build(seed, speed_factor)
