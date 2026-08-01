"""
maps/inflate.py

Grows every obstacle by the vehicle radius, once, when a map is built.

The planners and the scorer treat the vehicle as a point. Growing the obstacles by the vehicle's radius makes that point model honest: a route whose centre line clears the grown map clears the real map with the vehicle's body. The grown cells get their own value so a drawing can still show the buildings at their true size.
"""

import numpy as np
from scipy.ndimage import binary_dilation

# Cell value for the ring of cells the vehicle cannot stand on because its body would overlap an obstacle
MARGIN_VALUE = 4


def inflate_obstacles(grid, vehicle_radius):
    """
    Mark every free cell whose centre is within vehicle_radius of an obstacle cell as margin. The grid is edited in place.

    An obstacle cell is treated as a unit square and the vehicle as a circle standing on a cell centre, so the test is the distance from the centre to the nearest point of the square. This is the same overlap test the moving obstacle uses in hits(), so a static wall and the moving obstacle block the vehicle the same way.
    """
    # How many cells out the margin can possibly reach
    reach = int(np.ceil(vehicle_radius))

    # The set of offsets from an obstacle cell at which the vehicle's circle overlaps that cell's square
    structure = np.zeros((2 * reach + 1, 2 * reach + 1), dtype=bool)

    for change_y in range(-reach, reach + 1):
        for change_x in range(-reach, reach + 1):
            # Distance from the vehicle centre to the nearest edge of the square in each axis, zero when inside it
            gap_x = max(abs(change_x) - 0.5, 0.0)
            gap_y = max(abs(change_y) - 0.5, 0.0)

            if gap_x * gap_x + gap_y * gap_y <= vehicle_radius * vehicle_radius:
                structure[change_y + reach, change_x + reach] = True

    # Grow every obstacle by that set of offsets and mark only the new cells, so the obstacles keep their own values
    blocked = grid != 0
    grown = binary_dilation(blocked, structure=structure)
    grid[grown & ~blocked] = MARGIN_VALUE
