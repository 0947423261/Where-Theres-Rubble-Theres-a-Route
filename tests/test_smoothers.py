"""
tests/test_smoothers.py

Tests for the smoothing strategies.

Two things are worth checking about a smoother. It has to actually take the corners out, and it has to refuse to do so whenever the smooth version would cut through a building. The second matters more: a smoother that quietly shortens a route through a wall would improve every number in the results while making the route undriveable.
"""

import math
import unittest

import numpy as np

from planning.algorithms.smoothers import (
    BezierSmoother,
    BSplineSmoother,
    DubinsSmoother,
    NoSmoother,
    Smoother,
    _angle_gap,
    _dubins_LRL,
    _mod2pi,
)
from planning.utils.collision import validate_path
from planning.utils.config import Config
from planning.utils.metrics import curvature, max_curvature, path_length


def open_grid(size=40):
    return np.zeros((size, size), dtype=np.uint8)


def right_angle_path():
    """A path that drives along one axis and then turns square onto the other."""
    return [np.array([float(x), 5.0]) for x in range(5, 21)] + [
        np.array([20.0, float(y)]) for y in range(6, 21)
    ]


class TestCornerDetection(unittest.TestCase):
    def test_a_straight_line_has_no_corners(self):
        points = [np.array([float(x), 0.0]) for x in range(10)]
        self.assertEqual(Smoother._find_corners(points, 0.35), [])

    def test_a_right_angle_is_found_at_the_vertex(self):
        points = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0])]
        self.assertEqual(Smoother._find_corners(points, 0.35), [1])

    def test_a_gentle_bend_falls_under_the_threshold(self):
        # About six degrees of turn, which is the scale APF steps bend at and should be left alone
        points = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.1])]
        self.assertEqual(Smoother._find_corners(points, 0.35), [])

    def test_repeated_points_are_skipped_rather_than_counted(self):
        points = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0])]
        self.assertEqual(Smoother._find_corners(points, 0.35), [])

    def test_consecutive_corners_are_grouped_into_one_run(self):
        # A jagged stretch of RRT* is one run to smooth, not five separate corners
        self.assertEqual(Smoother._group_runs([3, 4, 5, 9, 10]), [[3, 4, 5], [9, 10]])

    def test_isolated_corners_stay_separate(self):
        self.assertEqual(Smoother._group_runs([2, 7, 15]), [[2], [7], [15]])


class TestCurveConstruction(unittest.TestCase):
    def test_a_bezier_passes_through_its_first_and_last_control_point(self):
        control = [np.array([0.0, 0.0]), np.array([2.0, 4.0]), np.array([6.0, 0.0])]
        curve = Smoother._bezier_curve(control, 20)

        np.testing.assert_allclose(curve[0], control[0], atol=1e-9)
        np.testing.assert_allclose(curve[-1], control[-1], atol=1e-9)

    def test_a_bezier_stays_inside_the_box_around_its_control_points(self):
        control = [np.array([0.0, 0.0]), np.array([2.0, 4.0]), np.array([6.0, 0.0])]
        curve = np.asarray(Smoother._bezier_curve(control, 30))

        # The curve lies in the convex hull of the control points, so it can never leave their bounding box
        self.assertTrue((curve[:, 0] >= -1e-9).all() and (curve[:, 0] <= 6 + 1e-9).all())
        self.assertTrue((curve[:, 1] >= -1e-9).all() and (curve[:, 1] <= 4 + 1e-9).all())

    def test_a_clamped_bspline_starts_and_ends_on_its_control_points(self):
        control = [np.array([float(i), float(i % 2)]) for i in range(6)]
        curve = Smoother._clamped_bspline(control, 20)

        np.testing.assert_allclose(curve[0], control[0], atol=1e-6)
        np.testing.assert_allclose(curve[-1], control[-1], atol=1e-6)

    def test_two_control_points_give_a_straight_line(self):
        control = [np.array([0.0, 0.0]), np.array([4.0, 0.0])]
        curve = np.asarray(Smoother._clamped_bspline(control, 10))

        # Two control points cannot support a cubic, so the degree drops to one and the curve is the line between them
        np.testing.assert_allclose(curve[:, 1], 0.0, atol=1e-9)
        np.testing.assert_allclose(curve[0], control[0], atol=1e-9)
        np.testing.assert_allclose(curve[-1], control[-1], atol=1e-9)


class TestDubinsHelpers(unittest.TestCase):
    def test_angles_are_wrapped_into_one_full_turn(self):
        for angle in (-7.0, -0.1, 0.0, 3.0, 9.0):
            self.assertTrue(0.0 <= _mod2pi(angle) < 2 * math.pi + 1e-12)

    def test_the_gap_between_two_headings_is_never_more_than_a_half_turn(self):
        # Two headings just either side of the wrap point are close together, not almost a full turn apart
        self.assertAlmostEqual(_angle_gap(0.1, 2 * math.pi - 0.1), 0.2, places=6)
        self.assertLessEqual(_angle_gap(0.0, math.pi + 1.0), math.pi)

    def test_the_left_right_left_word_runs_without_raising(self):
        # Regression test: this word tested an undefined name and raised instead of returning a result
        result = _dubins_LRL(0.3, 0.8, 1.2)
        self.assertTrue(result is None or len(result) == 4)

    def test_a_dubins_curve_lands_on_the_pose_it_was_aimed_at(self):
        start = (0.0, 0.0, 0.0)
        end = (10.0, 4.0, math.pi / 2)

        curve = Smoother._dubins_path(start, end, radius=2.5, step=0.4)

        self.assertIsNotNone(curve)
        np.testing.assert_allclose(curve[0], [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(curve[-1], [10.0, 4.0], atol=1e-9)

    def test_a_dubins_curve_never_turns_tighter_than_its_radius(self):
        curve = np.asarray(Smoother._dubins_path((0.0, 0.0, 0.0), (10.0, 4.0, math.pi / 2), 2.5, 0.4))

        # This is the guarantee that separates Dubins from the other two strategies, so it is worth measuring rather than assuming
        headings = np.arctan2(np.diff(curve[:, 1]), np.diff(curve[:, 0]))
        steps = np.linalg.norm(np.diff(curve, axis=0), axis=1)

        for turn, step in zip(np.abs(np.diff(np.unwrap(headings))), steps[1:]):
            if step > 1e-9:
                self.assertLessEqual(turn / step, 1.0 / 2.5 + 0.05)


class TestSmootherBehaviour(unittest.TestCase):
    def setUp(self):
        self.config = Config(out_dir="/tmp/test_outputs")
        self.grid = open_grid()
        self.path = right_angle_path()

    def strategies(self):
        return [
            BezierSmoother(self.config),
            BSplineSmoother(self.config),
            DubinsSmoother(self.config),
        ]

    def test_the_identity_strategy_changes_nothing(self):
        smoothed = NoSmoother(self.config).smooth(self.path, self.grid)
        self.assertIs(smoothed, self.path)

    def test_every_strategy_loosens_the_tightest_turn(self):
        for smoother in self.strategies():
            smoothed = smoother.smooth(list(self.path), self.grid)

            # Rounding a corner cannot change how much the route turns in total, since the directions it comes in and leaves on are fixed. What it changes is the tightest turn, which is what a vehicle with a minimum turning radius runs into
            self.assertLess(
                max_curvature(smoothed),
                max_curvature(self.path),
                f"{type(smoother).__name__} did not loosen the corner",
            )

    def test_the_dubins_strategy_respects_its_turning_radius(self):
        smoothed = DubinsSmoother(self.config).smooth(list(self.path), self.grid)

        # This is the guarantee the other two strategies do not give. A small margin is allowed for the arc being sampled into straight segments
        self.assertLessEqual(
            max_curvature(smoothed), 1.0 / self.config.dubins_radius + 0.05
        )

    def test_the_route_never_gets_longer_than_the_one_it_came_from(self):
        for smoother in self.strategies():
            smoothed = smoother.smooth(list(self.path), self.grid)

            # Cutting a corner shortens the route. A smoother that lengthened it would be adding detours rather than removing them
            self.assertLessEqual(
                path_length(smoothed), path_length(self.path) + 1e-6
            )

    def test_the_ordering_of_the_strategies_matches_what_they_promise(self):
        bezier = max_curvature(BezierSmoother(self.config).smooth(list(self.path), self.grid))
        bspline = max_curvature(BSplineSmoother(self.config).smooth(list(self.path), self.grid))
        dubins = max_curvature(DubinsSmoother(self.config).smooth(list(self.path), self.grid))

        # Dubins is bounded by its turning radius, Bezier cuts wide because every point of the window pulls on the whole curve, and the B-spline hugs the original path closely so it loosens the corner least
        self.assertLess(dubins, bezier)
        self.assertLess(bezier, bspline)

    def test_every_strategy_keeps_the_two_ends_where_they_were(self):
        for smoother in self.strategies():
            smoothed = smoother.smooth(list(self.path), self.grid)

            # Moving either end would change where the vehicle started or claim it arrived somewhere it did not
            np.testing.assert_allclose(smoothed[0], self.path[0], atol=1e-6)
            np.testing.assert_allclose(smoothed[-1], self.path[-1], atol=1e-6)

    def test_every_strategy_keeps_the_route_driveable(self):
        for smoother in self.strategies():
            smoothed = smoother.smooth(list(self.path), self.grid)
            self.assertTrue(validate_path(self.grid, smoothed))

    def test_a_corner_wrapped_around_a_building_is_left_alone(self):
        # The corner hugs the outside of a block, so any curve across it would cut the building. The smoother has to keep the sharp original rather than take the shortcut
        grid = open_grid()
        grid[6:20, 10:22] = 1

        path = [np.array([float(x), 5.0]) for x in range(2, 10)] + [
            np.array([9.0, float(y)]) for y in range(6, 22)
        ]

        for smoother in self.strategies():
            smoothed = smoother.smooth(list(path), grid)
            self.assertTrue(
                validate_path(grid, smoothed),
                f"{type(smoother).__name__} cut the corner through a building",
            )

    def test_a_path_too_short_to_have_a_corner_survives(self):
        for smoother in self.strategies():
            short = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
            smoothed = smoother.smooth(short, self.grid)
            self.assertEqual(len(smoothed), 2)


if __name__ == "__main__":
    unittest.main()
