"""
tests/test_optimal.py

The A* yardstick. If it is wrong, every length ratio is wrong in the same direction and no table would show it.
"""

import unittest

import numpy as np

from planning.utils.optimal import astar_length


def open_grid(size=20):
    return np.zeros((size, size), dtype=np.uint8)


class TestAStarLength(unittest.TestCase):
    def test_a_straight_line_costs_its_length_less_the_tolerance(self):
        # Ten cells apart, arrived within 1.5, so the shortest route stops after nine cells
        length = astar_length(open_grid(), (2, 5), (12, 5), goal_tolerance=1.5)
        self.assertAlmostEqual(length, 9.0)

    def test_a_diagonal_costs_root_two_per_step(self):
        length = astar_length(open_grid(), (2, 2), (6, 6), goal_tolerance=0.0)
        self.assertAlmostEqual(length, 4 * np.sqrt(2.0))

    def test_it_goes_around_a_wall(self):
        grid = open_grid()
        grid[2:18, 10] = 1

        straight = astar_length(open_grid(), (5, 10), (15, 10), goal_tolerance=0.0)
        around = astar_length(grid, (5, 10), (15, 10), goal_tolerance=0.0)

        self.assertIsNotNone(around)
        self.assertGreater(around, straight)

    def test_a_walled_off_goal_has_no_length(self):
        grid = open_grid()
        grid[:, 10] = 1
        self.assertIsNone(astar_length(grid, (5, 10), (15, 10), goal_tolerance=0.0))

    def test_a_diagonal_is_judged_by_the_same_segment_rule_as_a_planner_step(self):
        # A one cell diagonal samples its two ends and its midpoint, and the midpoint rounds onto one of the ends, so it is allowed past a corner exactly when a planner's step would be. Two free cells touching at a corner are therefore joined
        grid = open_grid()
        grid[5, 6] = 1
        grid[6, 5] = 1
        grid[4, 4:8] = 1
        grid[7, 4:8] = 1
        grid[4:8, 4] = 1
        grid[4:8, 7] = 1

        self.assertAlmostEqual(astar_length(grid, (5, 5), (6, 6), goal_tolerance=0.0), np.sqrt(2.0))

    def test_it_never_beats_the_straight_line(self):
        # The heuristic is admissible, so no route can come out shorter than the straight-line distance less the tolerance
        grid = open_grid(30)
        rng = np.random.default_rng(1)
        for _ in range(20):
            grid[rng.integers(3, 27), rng.integers(3, 27)] = 1

        length = astar_length(grid, (1, 1), (28, 27), goal_tolerance=1.5)
        self.assertGreaterEqual(length, np.hypot(27, 26) - 1.5)
