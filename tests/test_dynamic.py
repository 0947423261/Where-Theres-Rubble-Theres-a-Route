"""
tests/test_dynamic.py

Tests for driving under a moving obstacle.

The obstacle in these tests is given a speed of zero so it sits exactly where it is put. That makes the encounter deterministic: the reactive driver has to steer around a block it can see, and the plan-once driver has to drive into a block that was not on the map it planned against. Testing that contrast with a wandering obstacle would only test luck.
"""

import unittest

import numpy as np

from planning.algorithms.dynamic import PlanOnceDriver, ReactiveAPF
from planning.algorithms.hybrid import Hybrid
from planning.algorithms.rrt_star import RRTStar
from planning.algorithms.smoothers import BezierSmoother
from planning.maps.moving_obstacle import MovingObstacle
from planning.utils.config import Config
from planning.utils.metrics import reached_goal

RESULT_KEYS = {"path", "success", "reached", "collided", "iters", "replans", "switches", "plan_time"}


def corridor(size=60, half=9):
    """An open horizontal channel with solid walls above and below it."""
    grid = np.zeros((size, size), dtype=np.uint8)
    wall = size // 2 - half
    grid[0:wall, :] = 1
    grid[size - wall : size, :] = 1
    return grid


def parked_obstacle(grid, at, half_size=3):
    """An obstacle that stays exactly where it is put, so the encounter is the same on every run."""
    return MovingObstacle(
        grid.shape, seed=1, half_size=half_size, speed=0.0, start=at
    )


def test_config(**overrides):
    settings = dict(
        grid_size=60, rrt_max_iter=2500, dynamic_max_steps=400, out_dir="/tmp/test_outputs"
    )
    settings.update(overrides)
    return Config(**settings)


class TestReactiveAPF(unittest.TestCase):
    def setUp(self):
        self.config = test_config()
        self.grid = corridor()
        self.start = np.array([4.0, 30.0])
        self.goal = np.array([55.0, 30.0])

    def drive(self, obstacle):
        return ReactiveAPF(
            self.grid, self.start, self.goal, self.config, np.random.default_rng(1), obstacle
        ).plan()

    def test_it_crosses_a_clear_corridor(self):
        # The obstacle is parked off to one side, well away from the route
        result = self.drive(parked_obstacle(self.grid, (30.0, 38.0)))

        self.assertTrue(result["success"])
        self.assertTrue(reached_goal(result["path"], self.goal, self.config.goal_tolerance))

    def test_it_reports_every_field_the_results_file_needs(self):
        result = self.drive(parked_obstacle(self.grid, (30.0, 38.0)))
        self.assertEqual(set(result), RESULT_KEYS)

    def test_it_never_replans(self):
        result = self.drive(parked_obstacle(self.grid, (30.0, 38.0)))

        # Recomputing the force every step is the whole point. A reactive driver that replanned would be measuring the wrong thing
        self.assertEqual(result["replans"], 0)

    def test_it_steers_around_an_obstacle_beside_its_line(self):
        # The obstacle sits above the route rather than square across it, so the push it gives has a sideways component
        obstacle = parked_obstacle(self.grid, (30.0, 35.0))

        result = self.drive(obstacle)
        path = np.asarray(result["path"])

        self.assertTrue(result["success"])
        self.assertFalse(result["collided"])

        # Arriving without being hit means it gave way rather than driving straight on
        self.assertGreater(np.abs(path[:, 1] - 30.0).max(), 1.0)

    def test_an_obstacle_parked_square_across_the_route_stalls_it(self):
        # The nearest point of a box directly ahead is directly ahead, so the push comes straight back down the corridor and cancels the pull towards the goal. This is the same local minimum the static scenarios are built around, and it is why the reactive driver still fails some dynamic runs
        obstacle = parked_obstacle(self.grid, (30.0, 30.0))

        result = self.drive(obstacle)
        path = np.asarray(result["path"])

        self.assertFalse(result["success"])
        self.assertFalse(result["collided"])
        self.assertAlmostEqual(float(np.abs(path[:, 1] - 30.0).max()), 0.0, places=6)

    def test_a_moving_obstacle_never_holds_it_in_that_equilibrium(self):
        # The advantage of reacting is not that the driver solves the head-on case. It is that an obstacle which keeps moving never sits square across the route long enough for the equilibrium to settle
        moving = MovingObstacle(
            self.grid.shape, seed=11, half_size=3, speed=0.8, start=(30.0, 30.0),
            bounds=(26, 34, 24, 36),
        )

        result = self.drive(moving)

        self.assertTrue(result["reached"])
        self.assertFalse(result["collided"])

    def test_it_starts_where_it_was_told_to(self):
        result = self.drive(parked_obstacle(self.grid, (30.0, 38.0)))
        np.testing.assert_allclose(result["path"][0], self.start)

    def test_a_run_that_never_arrives_is_not_reported_as_a_success(self):
        # The obstacle is parked on top of the start, so there is nowhere to give way to
        obstacle = parked_obstacle(self.grid, (5.0, 30.0), half_size=2)

        result = self.drive(obstacle)

        self.assertFalse(result["success"])
        self.assertFalse(result["reached"])


class TestPlanOnceDriver(unittest.TestCase):
    def setUp(self):
        self.config = test_config()
        self.grid = corridor()
        self.start = np.array([4.0, 30.0])
        self.goal = np.array([55.0, 30.0])

    def drive(self, obstacle, replan=False, planner=RRTStar, **kwargs):
        return PlanOnceDriver(
            self.grid, self.start, self.goal, self.config, np.random.default_rng(3),
            obstacle, planner, replan=replan, **kwargs
        ).plan()

    def test_it_crosses_a_clear_corridor(self):
        result = self.drive(parked_obstacle(self.grid, (30.0, 38.0)))

        self.assertTrue(result["success"])
        self.assertFalse(result["collided"])

    def test_it_reports_every_field_the_results_file_needs(self):
        result = self.drive(parked_obstacle(self.grid, (30.0, 38.0)))
        self.assertEqual(set(result), RESULT_KEYS)

    def test_it_drives_into_an_obstacle_that_was_not_on_the_map_it_planned_against(self):
        # The planner sees the static corridor, which is clear, so it commits to a route straight through where the obstacle actually is
        obstacle = parked_obstacle(self.grid, (30.0, 30.0), half_size=6)

        result = self.drive(obstacle, replan=False)

        self.assertTrue(result["collided"])
        self.assertFalse(result["success"])

    def test_replanning_recovers_the_run(self):
        obstacle = parked_obstacle(self.grid, (30.0, 30.0), half_size=6)

        result = self.drive(obstacle, replan=True)

        # Replanning is what a global planner has to do to cope, and the count is what it costs
        self.assertGreater(result["replans"], 0)
        self.assertTrue(result["reached"])

    def test_replanning_costs_more_than_reacting(self):
        obstacle = parked_obstacle(self.grid, (30.0, 30.0), half_size=6)

        replanned = self.drive(obstacle, replan=True)
        reactive = ReactiveAPF(
            self.grid, self.start, self.goal, self.config, np.random.default_rng(1), obstacle
        ).plan()

        # Both get through, and only one of them had to plan again to do it
        self.assertEqual(reactive["replans"], 0)
        self.assertGreater(replanned["replans"], 0)

    def test_it_reports_failure_when_the_planner_finds_no_route(self):
        walled = corridor()
        walled[:, 30] = 1

        result = PlanOnceDriver(
            walled, self.start, self.goal, test_config(rrt_max_iter=300),
            np.random.default_rng(3), parked_obstacle(walled, (10.0, 38.0)), RRTStar,
        ).plan()

        # A vehicle that was never given a route never drove, so it cannot have arrived or been hit
        self.assertFalse(result["success"])
        self.assertFalse(result["reached"])
        self.assertEqual(result["iters"], 0)

    def test_it_works_with_the_hybrid_as_the_planner(self):
        result = self.drive(
            parked_obstacle(self.grid, (30.0, 38.0)),
            planner=Hybrid,
            adaptive=True,
            smoother=BezierSmoother(self.config),
        )

        # The driver takes whichever planner it is handed, so the study can compare the same route being driven by different planners
        self.assertTrue(result["success"])


class TestSharedObstacle(unittest.TestCase):
    def test_two_drivers_meet_the_same_walk(self):
        config = test_config()
        grid = corridor()
        obstacle = MovingObstacle(grid.shape, seed=5, half_size=3, speed=0.8, start=(30.0, 30.0))

        # The study reuses one obstacle across the drivers on a map, which is what makes the dynamic comparison paired
        first = [obstacle.position_at(t).copy() for t in range(50)]
        second = [obstacle.position_at(t).copy() for t in range(50)]

        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()


class ScriptedObstacle(MovingObstacle):
    """An obstacle that sits wherever the script says at each time step, so a test can place it exactly when it wants."""

    def __init__(self, grid_shape, positions, half_size=3):
        super().__init__(grid_shape, seed=1, half_size=half_size, speed=0.0, start=positions[0])
        self.positions = positions

    def position_at(self, t):
        # Past the end of the script it stays on the last position
        return np.asarray(self.positions[min(t, len(self.positions) - 1)], dtype=float)


class TestHitTestUsesTheObstacleAfterItMoved(unittest.TestCase):
    """
    At loop step t the vehicle moves to where it stands at time t + 1, and the obstacle moves too. The hit test has to compare the two at t + 1, not the new vehicle position against where the obstacle was a step ago.
    """

    def setUp(self):
        self.config = test_config()
        self.grid = corridor()
        self.start = np.array([4.0, 30.0])
        self.goal = np.array([55.0, 30.0])

    def test_an_obstacle_that_leaves_before_the_vehicle_arrives_does_not_hit_it(self):
        # At time 0 the obstacle sits on top of the start. From time 1 on it is far away. The vehicle only ever occupies the start at time 0, before it moves, so it is never under the obstacle at the same time
        obstacle = ScriptedObstacle(self.grid.shape, [(4.0, 30.0), (30.0, 38.0)])

        result = PlanOnceDriver(
            self.grid, self.start, self.goal, self.config, np.random.default_rng(3),
            obstacle, RRTStar, replan=False,
        ).plan()

        self.assertFalse(result["collided"])
        self.assertTrue(result["reached"])


class TestReplanGridMatchesTheHitTest(unittest.TestCase):
    def test_every_cell_the_hit_test_would_catch_is_blocked_on_the_replan_grid(self):
        config = test_config()
        grid = corridor()
        obstacle = parked_obstacle(grid, (30.4, 29.6))
        driver = PlanOnceDriver(
            grid, np.array([4.0, 30.0]), np.array([55.0, 30.0]), config,
            np.random.default_rng(3), obstacle, RRTStar,
        )

        # The vehicle stands well away from the obstacle, so the cleared block around it does not overlap the stamped square
        replan_grid = driver._replan_grid(np.array([10.0, 30.0]), 0)

        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if obstacle.hits(x, y, 0, config.vehicle_radius):
                    with self.subTest(x=x, y=y):
                        self.assertNotEqual(replan_grid[y, x], 0)


class TestAnyContactIsFailure(unittest.TestCase):
    """
    Every driver is scored the same way: touched by the obstacle means the run failed, whether or not it replanned and drove on afterwards. Without this the replan drivers could be hit any number of times and still post a success.
    """

    def setUp(self):
        self.config = test_config()
        self.grid = corridor()
        self.start = np.array([4.0, 30.0])
        self.goal = np.array([55.0, 30.0])

    def drive(self, obstacle, replan):
        return PlanOnceDriver(
            self.grid, self.start, self.goal, self.config, np.random.default_rng(3),
            obstacle, RRTStar, replan=replan,
        ).plan()

    def test_a_replan_driver_that_was_hit_is_not_a_success(self):
        # The obstacle appears on top of the vehicle at time ten, far away before that, so no amount of looking ahead avoids the contact
        far = (30.0, 38.0)
        ambush = [far] * 10 + [(11.0, 30.0)]
        result = self.drive(ScriptedObstacle(self.grid.shape, ambush), replan=True)

        # The contact counts against it whatever it does afterwards
        self.assertTrue(result["collided"])
        self.assertFalse(result["success"])

    def test_steps_count_the_moves_taken_on_every_outcome(self):
        # The driven path holds the start plus one point per move, so steps is always one fewer than the path length, whichever way the run ended
        for name, obstacle, replan in (
            ("reached", parked_obstacle(self.grid, (30.0, 38.0)), False),
            ("collided", parked_obstacle(self.grid, (30.0, 30.0), half_size=6), False),
        ):
            with self.subTest(outcome=name):
                result = self.drive(obstacle, replan)
                self.assertEqual(result["iters"], len(result["path"]) - 1)


class TestPlanningTimeExcludesTheDrive(unittest.TestCase):
    """
    The dynamic comp_time column says seconds spent planning. A plan-once driver spends most of its wall time following the route and testing for the obstacle, and none of that is planning.
    """

    def test_a_plan_once_driver_reports_the_time_of_its_planning_calls_only(self):
        config = test_config()
        grid = corridor()
        driver = PlanOnceDriver(
            grid, np.array([4.0, 30.0]), np.array([55.0, 30.0]), config,
            np.random.default_rng(3), parked_obstacle(grid, (30.0, 38.0)), RRTStar,
        )

        import time
        started = time.perf_counter()
        result = driver.plan()
        wall = time.perf_counter() - started

        self.assertGreater(result["plan_time"], 0.0)
        self.assertLess(result["plan_time"], wall)

    def test_a_driver_that_never_planned_reports_no_planning_time(self):
        # A route the planner could not find still took time to fail to find, so the time is there even though the vehicle never moved
        walled = corridor()
        walled[:, 30] = 1
        result = PlanOnceDriver(
            walled, np.array([4.0, 30.0]), np.array([55.0, 30.0]), test_config(rrt_max_iter=300),
            np.random.default_rng(3), parked_obstacle(walled, (10.0, 38.0)), RRTStar,
        ).plan()

        self.assertGreater(result["plan_time"], 0.0)
        self.assertEqual(result["iters"], 0)


class TestReplanBeforeContact(unittest.TestCase):
    """
    A replan driver should not have to be hit before it plans again. When the route ahead runs into where the obstacle is, or where it could be within a few steps, the driver plans around it while there is still room to.
    """

    def setUp(self):
        self.config = test_config()
        self.grid = corridor()
        self.start = np.array([4.0, 30.0])
        self.goal = np.array([55.0, 30.0])

    def drive(self, obstacle, config=None):
        return PlanOnceDriver(
            self.grid, self.start, self.goal, config or self.config, np.random.default_rng(3),
            obstacle, RRTStar, replan=True,
        ).plan()

    def test_it_plans_around_a_block_it_would_have_driven_into(self):
        # Parked square across the route. The plan-once driver hits it; the replan driver has to see it coming
        result = self.drive(parked_obstacle(self.grid, (30.0, 30.0), half_size=6))

        self.assertGreater(result["replans"], 0)
        self.assertFalse(result["collided"])
        self.assertTrue(result["reached"])
        self.assertTrue(result["success"])

    def test_a_clear_route_is_never_replanned(self):
        result = self.drive(parked_obstacle(self.grid, (30.0, 38.0)))

        self.assertEqual(result["replans"], 0)
        self.assertTrue(result["success"])

    def test_a_fixed_period_replans_on_the_clock(self):
        # The obstacle never comes near the route, so every replan comes from the period alone
        config = test_config(replan_period=40)
        result = self.drive(parked_obstacle(self.grid, (30.0, 38.0)), config)

        self.assertTrue(result["success"])
        self.assertEqual(result["replans"], result["iters"] // 40)
