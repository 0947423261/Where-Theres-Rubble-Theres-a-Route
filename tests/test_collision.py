"""
tests/test_collision.py

Tests for the boundary and collision helpers. Everything else in the project trusts these, so if they are wrong every result is wrong and nothing else would tell you.
"""

import unittest

import numpy as np

from planning.utils.collision import (
    checks_done,
    in_bounds,
    path_blocked,
    point_blocked,
    reset_checks,
    segment_blocked,
    validate_path,
)


def open_grid(size=10):
    """A square map with nothing on it."""
    return np.zeros((size, size), dtype=np.uint8)


class TestInBounds(unittest.TestCase):
    def test_inside_and_outside(self):
        grid = open_grid()

        # The corners of the map are inside it
        self.assertTrue(in_bounds(grid, 0, 0))
        self.assertTrue(in_bounds(grid, 9, 9))

        # One cell past any edge is outside it
        self.assertFalse(in_bounds(grid, -1, 0))
        self.assertFalse(in_bounds(grid, 10, 0))
        self.assertFalse(in_bounds(grid, 0, 10))


class TestPointBlocked(unittest.TestCase):
    def test_free_cell_is_not_blocked(self):
        self.assertFalse(point_blocked(open_grid(), 5, 5))

    def test_obstacle_cell_is_blocked(self):
        grid = open_grid()
        grid[5, 4] = 1
        # The grid is indexed row then column, so this is the point (x=4, y=5)
        self.assertTrue(point_blocked(grid, 4, 5))

    def test_any_non_zero_value_blocks(self):
        grid = open_grid()

        # Rubble and the moving obstacle use values 2 and 3 and have to block just as a building does
        grid[2, 2] = 2
        grid[3, 3] = 3
        self.assertTrue(point_blocked(grid, 2, 2))
        self.assertTrue(point_blocked(grid, 3, 3))

    def test_outside_the_map_is_blocked(self):
        # Driving off the map has to be refused the same way as driving into a wall
        self.assertTrue(point_blocked(open_grid(), -1, 5))
        self.assertTrue(point_blocked(open_grid(), 5, 40))

    def test_position_is_rounded_onto_a_cell(self):
        grid = open_grid()
        grid[5, 5] = 1

        # A position that rounds onto the blocked cell is blocked, and one that rounds off it is not
        self.assertTrue(point_blocked(grid, 5.4, 4.6))
        self.assertFalse(point_blocked(grid, 5.6, 4.4))


class TestSegmentBlocked(unittest.TestCase):
    def test_clear_segment(self):
        self.assertFalse(segment_blocked(open_grid(), (1, 1), (8, 8)))

    def test_segment_through_a_wall(self):
        grid = open_grid()

        # A wall down the middle of the map with no gap in it
        grid[:, 5] = 1
        self.assertTrue(segment_blocked(grid, (1, 1), (9, 1)))

    def test_segment_along_a_wall_is_clear(self):
        grid = open_grid()
        grid[:, 5] = 1

        # Driving alongside the wall never touches it
        self.assertFalse(segment_blocked(grid, (2, 1), (2, 8)))

    def test_a_wall_thinner_than_the_sampling_is_still_caught(self):
        grid = open_grid(30)
        grid[:, 15] = 1

        # The default sampling is half a cell, so a wall one cell thick cannot be stepped over
        self.assertTrue(segment_blocked(grid, (1, 1), (29, 1)))


class TestPathBlocked(unittest.TestCase):
    def test_clear_path(self):
        self.assertFalse(path_blocked(open_grid(), [(1, 1), (4, 4), (8, 8)]))

    def test_path_whose_middle_segment_crosses_a_wall(self):
        grid = open_grid()
        grid[:, 5] = 1

        # Both ends sit on free cells, and only the segment between them is blocked
        self.assertTrue(path_blocked(grid, [(1, 1), (9, 1)]))


class TestValidatePath(unittest.TestCase):
    def test_clear_path_is_valid(self):
        self.assertTrue(validate_path(open_grid(), [(1, 1), (4, 4), (8, 8)]))

    def test_path_through_an_obstacle_is_rejected(self):
        grid = open_grid()
        grid[:, 5] = 1
        self.assertFalse(validate_path(grid, [(1, 1), (9, 1)]))

    def test_path_with_a_point_inside_an_obstacle_is_rejected(self):
        grid = open_grid()
        grid[4, 4] = 1
        self.assertFalse(validate_path(grid, [(1, 1), (4, 4)]))

    def test_empty_path_is_rejected(self):
        # Nothing to drive along is not the same as a valid route
        self.assertFalse(validate_path(open_grid(), []))
        self.assertFalse(validate_path(open_grid(), None))

    def test_path_leaving_the_map_is_rejected(self):
        self.assertFalse(validate_path(open_grid(), [(1, 1), (20, 20)]))


if __name__ == "__main__":
    unittest.main()


class TestCheckCounter(unittest.TestCase):
    """
    Every planner is charged one check per cell it asks about. The counter is what the budget column in the results is read from, so it has to count every point test and nothing else.
    """

    def test_it_starts_at_zero_after_a_reset(self):
        point_blocked(open_grid(), 1, 1)
        reset_checks()
        self.assertEqual(checks_done(), 0)

    def test_every_point_test_counts_one(self):
        reset_checks()
        grid = open_grid()
        for i in range(5):
            point_blocked(grid, i, i)
        self.assertEqual(checks_done(), 5)

    def test_a_segment_counts_one_per_sample(self):
        reset_checks()

        # Four cells long at half a cell spacing is nine samples, the two ends included
        segment_blocked(open_grid(), (1.0, 1.0), (5.0, 1.0), spacing=0.5)
        self.assertEqual(checks_done(), 9)

    def test_a_blocked_segment_stops_counting_where_it_stops_looking(self):
        reset_checks()
        grid = open_grid()
        grid[1, 3] = 1

        # Samples at x = 1, 1.5, 2, 2.5 and 3. Python rounds 2.5 down to the even cell 2, so the fifth sample is the first to land on cell 3 and be blocked. Nothing past it is looked at
        segment_blocked(grid, (1.0, 1.0), (5.0, 1.0), spacing=0.5)
        self.assertEqual(checks_done(), 5)
