"""
maps/scenarios.py

Random map generators for the three static scenarios

Each map is a post-disaster street grid: solid building blocks separated by streets, with piles of rubble dropped into the streets on top. The vehicle drives the street network and the rubble is what turns a clear route into a trap.

Cell values used across the whole project:
    0 = free street or open ground, driveable
    1 = intact building
    2 = rubble from a collapse
    3 = the moving obstacle (only ever stamped in by maps/moving_obstacle.py)
    4 = the margin around every obstacle that the vehicle's body cannot enter (added by maps/inflate.py)

The planners treat every non-zero value the same way because collision.py only asks whether a cell is non-zero. The separate values exist so the figures and animations can colour buildings, rubble and the moving obstacle differently, and draw the buildings at their true size with the margin faint around them.

The three scenarios are:
    normal_city  - wide streets, intact blocks and a little light debris
    blocked_road - the same city with a U-shaped collapse straddling the direct route, its opening facing the start, which is the textbook trap that holds APF in a local minimum
    dense_city   - narrow streets, heavy rubble and a few streets sealed end to end, which produces dead ends that were not placed by hand

APF is deterministic, so the variation across a scenario has to come from the maps rather than from the algorithm. Every scenario is generated from a seed, and the same seed always rebuilds the same map, so all the algorithms can be measured on one shared set of instances.

Every map is checked with a flood fill before it is handed out, so a failure in the results is the fault of the algorithm and never of an impossible map.
"""

from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from .inflate import inflate_obstacles


class Scenario(ABC):
    """Abstract base for the scenario generators. It owns the shared carving and rubble helpers, and each subclass only decides how its own map is populated."""

    # Constructor just holds the configuration
    def __init__(self, config):
        self.config = config

    def build(self, seed):
        """
        Build one map from the seed and return (grid, start, goal).

        The generator retries with a nudged seed while the goal is walled off, so the caller always receives a solvable map.
        """
        # The start and goal come from the configuration and are the same for every instance, so only the obstacles vary
        start = np.array(self.config.start, dtype=float)
        goal = np.array(self.config.goal, dtype=float)

        for attempt in range(30):
            # A fresh generator per attempt, offset far enough that a retry is a genuinely different map rather than a shifted version of the last one
            rng = np.random.default_rng(seed + attempt * 1000)

            # Start from an empty map of the configured size, stored as small unsigned integers since the cell values only go up to 3
            grid = np.zeros(
                (self.config.grid_size, self.config.grid_size), dtype=np.uint8
            )

            # Let the subclass lay out its own streets, buildings and debris
            self._populate(grid, rng, start, goal)

            # The vehicle has to be able to stand at both ends, so clear a small square around each
            self._clear_around(grid, int(start[0]), int(start[1]), radius=2)
            self._clear_around(grid, int(goal[0]), int(goal[1]), radius=2)

            # Grow every obstacle by the vehicle radius, so the planners and the scorer see the map the vehicle's body has to fit through. This has to come before the reachability check, or a gap the body cannot fit through would pass as a route
            inflate_obstacles(grid, self.config.vehicle_radius)

            # Only hand back a map where a route actually exists
            if self._reachable(grid, start, goal):
                return grid, start, goal

        raise RuntimeError(
            f"Could not build a solvable '{type(self).__name__}' map from seed {seed} after 30 attempts."
        )

    @abstractmethod
    def _populate(self, grid, rng, start, goal):
        """
        Fill the empty grid with this scenario's buildings and debris. The grid is edited in place.
        """
        raise NotImplementedError

    ### === Shared building blocks === ###

    @staticmethod
    def _carve_street_grid(grid, rng, street_width, block_min, block_max):
        """
        Fill the map with solid building and then cut a lattice of streets back out of it.

        The block sizes between streets are drawn at random between block_min and block_max, so the city is a regular street network without being a perfect chessboard.
        """
        # Grid dimensions for the loop bounds
        h, w = grid.shape

        # Every cell starts as building and the streets are carved out of it
        grid[:, :] = 1

        # Horizontal streets. Carve one street, skip a randomly sized block, repeat down the map
        y = 0
        while y < h:
            grid[y : y + street_width, :] = 0
            y += street_width + int(rng.integers(block_min, block_max + 1))

        # Vertical streets, carved the same way across the map
        x = 0
        while x < w:
            grid[:, x : x + street_width] = 0
            x += street_width + int(rng.integers(block_min, block_max + 1))

        # A street along each of the four edges, which keeps the corners where the start and goal live on driveable ground
        grid[0:street_width, :] = 0
        grid[h - street_width : h, :] = 0
        grid[:, 0:street_width] = 0
        grid[:, w - street_width : w] = 0

    @staticmethod
    def _drop_rubble_pile(grid, rng, n_cells):
        """
        Drop one pile of rubble into the streets. The pile starts at a random free cell and grows along a random walk, so it spreads along the street and banks up against the buildings rather than forming a neat rectangle.

        Rubble only ever lands on free cells, so a collapse never eats into a building that is still standing.
        """
        # Grid dimensions for clamping the walk
        h, w = grid.shape

        # Look for a free cell to start the pile on
        for _ in range(200):
            x = int(rng.integers(1, w - 1))
            y = int(rng.integers(1, h - 1))
            if grid[y, x] == 0:
                break
        else:
            # The map has no free cell to start from, so there is nothing to drop
            return

        # The four directions the walk can take on each step
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for _ in range(n_cells):
            # Only free cells become rubble, so the pile flows around standing buildings
            if grid[y, x] == 0:
                grid[y, x] = 2

            # Take one step in a random direction, clamped so the walk stays inside the map
            change_x, change_y = directions[int(rng.integers(0, 4))]
            x = int(np.clip(x + change_x, 1, w - 2))
            y = int(np.clip(y + change_y, 1, h - 2))

    @staticmethod
    def _close_street(grid, rng):
        """
        Seal one street from wall to wall with rubble, as though a facade had come down across it. This is what creates the dead ends in the dense scenario.

        The wall is grown out from a random free cell in one axis until it reaches a building on both sides. It has to run across the street and not along it: along a street the only thing that stops it is the far edge of the map, and a wall like that cuts the map in two. Before the vehicle margin the edge row of the map slipped past that wall, which is the only reason those maps ever came out solvable. So a run of cells is only used when it is short enough to be one street, and a cell where both axes run long is skipped.
        """
        # Grid dimensions for the bounds checks
        h, w = grid.shape

        # The two ways the wall can run, as (change_x, change_y) unit steps: up and down, then left and right
        axes = ((0, 1), (1, 0))

        # The longest run that can still be one street with an intersection behind it. Anything longer has run along a street
        longest = 12

        for _ in range(200):
            # Pick a candidate cell away from the very edge of the map
            x = int(rng.integers(3, w - 3))
            y = int(rng.integers(3, h - 3))

            # The wall has to start in the street rather than inside a building
            if grid[y, x] != 0:
                continue

            # Half the time try up and down first, the rest of the time left and right, so an intersection is not always sealed the same way
            first = int(rng.random() < 0.5)

            for change_x, change_y in (axes[first], axes[1 - first]):
                # Gather the run of non-building cells through the candidate along this axis, stopping at a building or the map edge
                cells = [(x, y)]
                for step in (1, -1):
                    check_x, check_y = x + change_x * step, y + change_y * step
                    while 0 < check_x < w - 1 and 0 < check_y < h - 1 and grid[check_y, check_x] != 1:
                        cells.append((check_x, check_y))
                        check_x += change_x * step
                        check_y += change_y * step

                # A long run means this axis lies along the street, so try the other one
                if len(cells) > longest:
                    continue

                # Rubble only lands on free cells, so an existing pile is left as it is
                for check_x, check_y in cells:
                    if grid[check_y, check_x] == 0:
                        grid[check_y, check_x] = 2

                # One closure per call, so the caller controls how many there are
                return

    @staticmethod
    def _collapsed_building_trap(grid, rng, start, goal):
        """
        Build the U-shaped collapse that gives the blocked road scenario its name. The back wall sits across the straight line from the start to the goal and the two arms reach back towards the start, so the opening faces the vehicle.

        Inside the pocket the pull towards the goal and the push off the back wall cancel, which is exactly the local minimum that standalone APF cannot leave. A route around the outside always survives because the whole map is checked for reachability afterwards.
        """
        # Grid dimensions for clipping the rectangles
        h, w = grid.shape

        # Centre the trap near the midpoint of the straight line between the start and the goal, jittered so it is not in the same place on every instance
        center_x = int((start[0] + goal[0]) / 2) + int(rng.integers(-3, 4))
        center_y = int((start[1] + goal[1]) / 2) + int(rng.integers(-3, 4))

        # Half the width of the U, how deep the pocket runs and how thick the debris walls are. The first two scale with the map so the trap is the same shape on a test-sized grid, a fifth of the width across and an eighth deep
        arm = w // 5
        depth = h // 8
        thickness = 2

        def fill(x0, x1, y0, y1):
            """Turn the free cells of one rectangle into rubble, leaving standing buildings alone."""
            # Clip the rectangle to the map before slicing
            x0, x1 = max(0, x0), min(w, x1)
            y0, y1 = max(0, y0), min(h, y1)
            region = grid[y0:y1, x0:x1]
            region[region == 0] = 2

        # The back wall, which is the far side of the pocket and blocks the way to the goal
        fill(center_x - arm, center_x + arm + 1, center_y + depth, center_y + depth + thickness)

        # The left arm reaching back towards the start
        fill(center_x - arm, center_x - arm + thickness, center_y - depth, center_y + depth + thickness)

        # The right arm, which closes the pocket on the other side
        fill(center_x + arm - thickness + 1, center_x + arm + 1, center_y - depth, center_y + depth + thickness)

        # The pocket between the arms and in front of the back wall, as (x_min, x_max, y_min, y_max) inclusive, so a test can ask whether a point is inside the trap
        return (
            center_x - arm + thickness,
            center_x + arm - thickness,
            center_y - depth,
            center_y + depth - 1,
        )

    @staticmethod
    def _clear_around(grid, x, y, radius):
        """Force a small square around (x, y) to be free, which is used to keep the start and the goal from being buried."""
        # Grid dimensions for clipping the square
        h, w = grid.shape

        # Clip the square to the map, remembering that the upper bound of a slice is exclusive
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)

        grid[y0:y1, x0:x1] = 0

    @staticmethod
    def _reachable(grid, start, goal):
        """
        Breadth-first flood fill from the start across free cells. Returns True if the fill arrives next to the goal, meaning at least one route exists.
        """
        # Grid dimensions for the bounds test
        h, w = grid.shape

        # Round both ends onto the cells they sit in
        start_x, start_y = int(round(start[0])), int(round(start[1]))
        goal_x, goal_y = int(round(goal[0])), int(round(goal[1]))

        # If either end is buried there is nothing to search
        if grid[start_y, start_x] != 0 or grid[goal_y, goal_x] != 0:
            return False

        # Cells already added to the queue, so the fill never revisits one
        seen = np.zeros(grid.shape, dtype=bool)
        seen[start_y, start_x] = True

        # The frontier of the fill, popped from the left so the search stays breadth first
        queue = deque([(start_x, start_y)])

        while queue:
            x, y = queue.popleft()

            # Arriving in the ring of cells around the goal is close enough, since the goal tolerance is larger than one cell
            if abs(x - goal_x) <= 1 and abs(y - goal_y) <= 1:
                return True

            # Spread into the four neighbouring cells
            for change_x, change_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_x, next_y = x + change_x, y + change_y

                # Only free, in-bounds cells that have not been seen join the frontier
                if (
                    0 <= next_x < w
                    and 0 <= next_y < h
                    and not seen[next_y, next_x]
                    and grid[next_y, next_x] == 0
                ):
                    seen[next_y, next_x] = True
                    queue.append((next_x, next_y))

        # The queue emptied without reaching the goal, so the goal is walled off
        return False


class NormalCity(Scenario):
    """An intact city with wide streets, large blocks and a scattering of light debris. This is the easy scenario where APF should do well on its own."""

    def _populate(self, grid, rng, start, goal):
        # Wide streets and large blocks leave plenty of room to drive. Eight cells is six once the vehicle margin is on, which is enough for a vehicle two cells wide to take a corner at the Dubins radius
        self._carve_street_grid(grid, rng, street_width=8, block_min=13, block_max=19)

        # Two to four small piles of debris, which narrow a few streets without closing any
        for _ in range(int(rng.integers(2, 5))):
            self._drop_rubble_pile(grid, rng, n_cells=int(rng.integers(6, 13)))


class BlockedRoad(Scenario):
    """The same intact city with a collapsed building forming a U-shaped trap across the direct route. This is the scenario that separates the hybrid from standalone APF."""

    # The pocket of the last map built, as (x_min, x_max, y_min, y_max) inclusive. None until build has run
    pocket = None

    def _populate(self, grid, rng, start, goal):
        # The same street layout as the normal city, so the trap is the only difference between the two scenarios
        self._carve_street_grid(grid, rng, street_width=8, block_min=13, block_max=19)

        # The collapse that straddles the straight line to the goal. Its pocket is kept so the escape tests can tell whether a sub-goal landed inside it
        self.pocket = self._collapsed_building_trap(grid, rng, start, goal)

        # A little extra debris so the trap is not the only obstacle on the map
        for _ in range(int(rng.integers(2, 4))):
            self._drop_rubble_pile(grid, rng, n_cells=int(rng.integers(6, 12)))


class DenseCity(Scenario):
    """A heavily damaged dense city with narrow streets, heavy debris and a few streets sealed end to end. This is the hard scenario where dead ends appear on their own."""

    def _populate(self, grid, rng, start, goal):
        # Narrow streets and tight blocks, which leaves a much smaller driveable channel. With the vehicle margin on, a six cell street is four cells wide, two vehicles abreast and no room to turn at the Dubins radius
        self._carve_street_grid(grid, rng, street_width=6, block_min=10, block_max=15)

        # Eight to twelve piles of debris spread across the street network
        for _ in range(int(rng.integers(8, 13))):
            self._drop_rubble_pile(grid, rng, n_cells=int(rng.integers(7, 18)))

        # Two or three streets sealed from wall to wall, which is what produces the dead ends
        for _ in range(int(rng.integers(2, 4))):
            self._close_street(grid, rng)


# Maps the scenario names used in the configuration and in the results to the class that builds them
SCENARIOS = {
    "normal_city": NormalCity,
    "blocked_road": BlockedRoad,
    "dense_city": DenseCity,
}


def make_scenario(scenario, seed, config):
    """
    Build one map of the named scenario from the given seed and return (grid, start, goal, generator). The generator is handed back too because the blocked road one remembers where its pocket is, which the detection-time measure needs.
    """
    # Fail loudly on a name that was never registered rather than silently building the wrong map
    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Choose one of {list(SCENARIOS)}."
        )

    generator = SCENARIOS[scenario](config)
    grid, start, goal = generator.build(seed)

    return grid, start, goal, generator


def make_instance(scenario, seed, config):
    """
    Build one map of the named scenario from the given seed and return (grid, start, goal).

    This is the single entry point the runners and the animation scripts use, so every part of the project builds instance i of a scenario the same way.
    """
    grid, start, goal, _ = make_scenario(scenario, seed, config)
    return grid, start, goal
