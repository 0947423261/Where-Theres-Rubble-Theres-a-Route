"""
tests/test_metrics.py

Tests for the measurements every algorithm is scored on.

Each one is checked against a path where the right answer can be worked out by hand, so a change to the measurement that quietly flatters one algorithm shows up here rather than in the results chapter.
"""

import math
import unittest

import numpy as np

from planning.utils.config import Config
from planning.utils.metrics import (
    clearance,
    curvature,
    drive_time,
    path_length,
    radius_violated,
    reached_goal,
    score_run,
    sharp_turns,
    smoothness,
    time_to_detection,
    total_turning,
)


def open_grid(size=20):
    return np.zeros((size, size), dtype=np.uint8)


class TestPathLength(unittest.TestCase):
    def test_a_straight_line(self):
        # Three points a fixed distance apart on one axis
        self.assertAlmostEqual(path_length([(0, 0), (3, 0), (7, 0)]), 7.0)

    def test_a_diagonal_uses_the_straight_line_distance(self):
        # The hypotenuse of a three four five triangle
        self.assertAlmostEqual(path_length([(0, 0), (3, 4)]), 5.0)

    def test_a_path_too_short_to_have_a_length(self):
        self.assertEqual(path_length([(1, 1)]), 0.0)
        self.assertEqual(path_length(None), 0.0)


class TestSmoothness(unittest.TestCase):
    def test_a_straight_line_never_turns(self):
        self.assertAlmostEqual(smoothness([(0, 0), (1, 0), (2, 0), (3, 0)]), 0.0)

    def test_reversing_at_every_point_is_the_worst_case(self):
        # The path doubles back on itself, which is a half turn at each interior point
        self.assertAlmostEqual(smoothness([(0, 0), (1, 0), (0, 0), (1, 0)]), 1.0)

    def test_a_right_angle_is_half_of_a_half_turn(self):
        # One quarter turn out of a possible half turn at the single interior point
        self.assertAlmostEqual(smoothness([(0, 0), (1, 0), (1, 1)]), 0.5)

    def test_the_measure_falls_when_the_same_corner_is_sampled_more_finely(self):
        # The same right angle, once as three points and once walked in single cell steps. Dividing by the number of vertices means the finer sampling scores better even though the route is identical, which is why curvature is reported alongside it
        coarse = smoothness([(0, 0), (4, 0), (4, 4)])
        fine = smoothness([(x, 0) for x in range(5)] + [(4, y) for y in range(1, 5)])

        self.assertGreater(coarse, fine)

    def test_a_path_too_short_to_bend(self):
        self.assertEqual(smoothness([(0, 0), (1, 1)]), 0.0)

    def test_repeated_points_are_ignored_rather_than_counted_as_turns(self):
        # A planner that stands still for a step has not turned
        self.assertAlmostEqual(smoothness([(0, 0), (1, 0), (1, 0), (2, 0)]), 0.0)


class TestCurvature(unittest.TestCase):
    def test_a_straight_line_has_no_curvature(self):
        self.assertAlmostEqual(curvature([(0, 0), (1, 0), (2, 0), (3, 0)]), 0.0)

    def test_it_does_not_depend_on_how_finely_the_path_is_sampled(self):
        # The same right angle sampled two ways. This is the property smoothness does not have, and the reason this measure exists
        coarse = curvature([(0, 0), (4, 0), (4, 4)])
        fine = curvature([(x, 0) for x in range(5)] + [(4, y) for y in range(1, 5)])

        self.assertAlmostEqual(coarse, fine, places=6)

    def test_a_right_angle_is_a_quarter_turn_spread_over_the_distance_driven(self):
        # A quarter turn over eight cells of driving
        self.assertAlmostEqual(curvature([(0, 0), (4, 0), (4, 4)]), (math.pi / 2) / 8.0)

    def test_a_tighter_turn_over_the_same_distance_scores_worse(self):
        gentle = curvature([(0, 0), (4, 0), (8, 1)])
        sharp = curvature([(0, 0), (4, 0), (4, 4)])

        self.assertGreater(sharp, gentle)


class TestClearance(unittest.TestCase):
    def test_distance_from_a_single_obstacle(self):
        grid = open_grid()
        grid[10, 10] = 1

        # A point five cells away from the obstacle, measured along one axis
        minimum, mean = clearance(grid, [(5, 10)])
        self.assertAlmostEqual(minimum, 5.0)
        self.assertAlmostEqual(mean, 5.0)

    def test_the_minimum_is_the_closest_the_path_ever_came(self):
        grid = open_grid()
        grid[10, 10] = 1

        minimum, mean = clearance(grid, [(5, 10), (8, 10)])
        self.assertAlmostEqual(minimum, 2.0)
        self.assertGreater(mean, minimum)

    def test_the_margin_is_not_an_obstacle(self):
        # The planners see the grown obstacle, but clearance is the gap to the real wall. A point one cell off the margin is two cells off the wall
        grid = open_grid()
        grid[10, 10] = 1
        grid[9:12, 9:12][grid[9:12, 9:12] == 0] = 4

        minimum, _ = clearance(grid, [(8, 10)])
        self.assertAlmostEqual(minimum, 2.0)

    def test_it_reads_between_the_cells(self):
        # Half way between two cells reads half way between their distances, so a route is not snapped onto the grid before it is measured
        grid = open_grid()
        grid[10, 10] = 1

        minimum, _ = clearance(grid, [(5.5, 10)])
        self.assertAlmostEqual(minimum, 4.5)

    def test_a_point_on_an_obstacle_has_no_clearance(self):
        grid = open_grid()
        grid[10, 10] = 1

        minimum, _ = clearance(grid, [(10, 10)])
        self.assertAlmostEqual(minimum, 0.0)


class TestReachedGoal(unittest.TestCase):
    def test_arriving_inside_the_tolerance(self):
        self.assertTrue(reached_goal([(0, 0), (10, 10)], (10, 11), tolerance=1.5))

    def test_stopping_outside_the_tolerance(self):
        self.assertFalse(reached_goal([(0, 0), (10, 10)], (10, 15), tolerance=1.5))

    def test_no_path_never_arrives(self):
        self.assertFalse(reached_goal(None, (1, 1), tolerance=1.5))


class TestScoreRun(unittest.TestCase):
    def setUp(self):
        self.config = Config(out_dir="/tmp/test_outputs")

    def test_a_clean_run_is_scored_successful(self):
        grid = open_grid()
        path = [(1, 1), (5, 5), (9, 9)]

        scores = score_run(grid, path, (9, 9), self.config, reported_success=True)

        self.assertTrue(scores["success"])
        self.assertAlmostEqual(scores["path_length"], path_length(path))

        # Every measurement the results file has a column for has to come back
        for key in ("smoothness", "curvature", "clearance_min", "clearance_mean"):
            self.assertIn(key, scores)

    def test_a_planner_that_claims_success_without_arriving_is_not_believed(self):
        grid = open_grid()

        scores = score_run(grid, [(1, 1), (5, 5)], (9, 9), self.config, reported_success=True)

        # The independent arrival check is what stops a planner marking its own homework
        self.assertFalse(scores["success"])

    def test_a_path_that_cuts_through_a_wall_is_not_successful(self):
        grid = open_grid()
        grid[:, 5] = 1

        # This is the check that caught RRT* growing branches through buildings
        scores = score_run(grid, [(1, 1), (9, 1)], (9, 1), self.config, reported_success=True)
        self.assertFalse(scores["success"])

    def test_a_failed_run_is_still_scored_rather_than_dropped(self):
        scores = score_run(open_grid(), None, (9, 9), self.config, reported_success=False)

        # Every run needs a row, otherwise the success rate is computed over a different number of runs per algorithm
        self.assertFalse(scores["success"])
        self.assertEqual(scores["path_length"], 0.0)


if __name__ == "__main__":
    unittest.main()


def staircase():
    """Four unit steps alternating right and up, so three right-angle corners."""
    return [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]]


def arc(radius, n_points=12):
    """A quarter circle of the given radius sampled about one cell apart, so the turning is spread and the curvature is 1 / radius."""
    angles = np.linspace(0.0, math.pi / 2, n_points)
    return [[radius * math.cos(a), radius * math.sin(a)] for a in angles]


class TestTotalTurning(unittest.TestCase):
    def test_three_right_angles_add_up(self):
        angle, count = total_turning(staircase())
        self.assertAlmostEqual(angle, 3 * math.pi / 2)
        self.assertEqual(count, 3)

    def test_a_straight_line_turns_nowhere(self):
        angle, _ = total_turning([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        self.assertAlmostEqual(angle, 0.0)


class TestSharpTurns(unittest.TestCase):
    def test_every_right_angle_is_sharp(self):
        self.assertEqual(sharp_turns(staircase(), threshold=0.35), 3)

    def test_a_gentle_arc_has_none(self):
        # A quarter turn over twelve points is about 0.14 radians per vertex, under the threshold at every one
        self.assertEqual(sharp_turns(arc(8.0), threshold=0.35), 0)

    def test_the_threshold_is_exclusive(self):
        # Exactly the threshold is not sharper than it
        self.assertEqual(sharp_turns(staircase(), threshold=math.pi / 2), 0)

    def test_a_path_too_short_to_turn(self):
        self.assertEqual(sharp_turns([[0.0, 0.0], [1.0, 0.0]], threshold=0.35), 0)


class TestRadiusViolated(unittest.TestCase):
    def test_a_right_angle_is_tighter_than_any_useful_radius(self):
        self.assertTrue(radius_violated(staircase(), min_radius=2.5))

    def test_an_arc_wider_than_the_radius_is_fine(self):
        self.assertFalse(radius_violated(arc(8.0), min_radius=2.5))

    def test_an_arc_tighter_than_the_radius_is_not(self):
        self.assertTrue(radius_violated(arc(1.5), min_radius=2.5))

    def test_a_straight_line_never_violates(self):
        self.assertFalse(radius_violated([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], min_radius=2.5))


class TestScoreRunReportsTheTurningColumns(unittest.TestCase):
    def test_the_three_columns_are_in_the_scores(self):
        config = Config(out_dir="/tmp/test_outputs")
        scores = score_run(open_grid(), staircase(), goal=[2.0, 2.0], config=config, reported_success=True)

        self.assertAlmostEqual(scores["total_turning"], 3 * math.pi / 2)
        self.assertEqual(scores["sharp_turns"], 3)
        self.assertTrue(scores["radius_violated"])


class TestDriveTime(unittest.TestCase):
    def test_a_straight_line_drives_at_full_speed(self):
        straight = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]
        self.assertAlmostEqual(drive_time(straight, speed=8.3, lateral_accel=2.0), 20.0 / 8.3, places=6)

    def test_a_zigzag_of_the_same_length_takes_longer(self):
        # Twenty cells driven either way. The zigzag turns a right angle every cell, so it is held at the speed the turns allow
        straight = [[0.0, 0.0], [20.0, 0.0]]
        zigzag = [[0.0, 0.0]]
        for i in range(20):
            step = [1.0, 0.0] if i % 2 == 0 else [0.0, 1.0]
            zigzag.append([zigzag[-1][0] + step[0], zigzag[-1][1] + step[1]])

        self.assertAlmostEqual(path_length(straight), path_length(zigzag))
        self.assertGreater(drive_time(zigzag, 8.3, 2.0), drive_time(straight, 8.3, 2.0))

    def test_a_gentler_arc_is_driven_faster(self):
        self.assertLess(
            drive_time(arc(8.0), 8.3, 2.0) / path_length(arc(8.0)),
            drive_time(arc(1.5), 8.3, 2.0) / path_length(arc(1.5)),
        )

    def test_it_is_in_the_scores(self):
        config = Config(out_dir="/tmp/test_outputs")
        scores = score_run(open_grid(), staircase(), goal=[2.0, 2.0], config=config, reported_success=True)
        self.assertGreater(scores["drive_time_curved"], scores["path_length"] / config.vehicle_speed)


class TestTimeToDetection(unittest.TestCase):
    # A pocket from x 10 to 20, y 10 to 20, inclusive
    pocket = (10, 20, 10, 20)

    def path_into_pocket(self):
        # Five points outside, entering at index 5, then five more inside
        return [[float(x), 15.0] for x in range(5, 16)]

    def test_steps_from_entry_to_the_first_switch(self):
        # Entered at index 5, switched at index 9: four steps of not noticing
        self.assertEqual(time_to_detection(self.path_into_pocket(), [9, 30], self.pocket), 4)

    def test_a_switch_before_entering_is_not_a_detection(self):
        self.assertTrue(np.isnan(time_to_detection(self.path_into_pocket(), [2], self.pocket)))

    def test_no_switch_is_no_detection(self):
        self.assertTrue(np.isnan(time_to_detection(self.path_into_pocket(), [], self.pocket)))

    def test_never_entering_is_no_detection(self):
        outside = [[float(x), 30.0] for x in range(5, 16)]
        self.assertTrue(np.isnan(time_to_detection(outside, [8], self.pocket)))

    def test_no_pocket_is_no_detection(self):
        self.assertTrue(np.isnan(time_to_detection(self.path_into_pocket(), [9], None)))
