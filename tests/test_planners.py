"""
tests/test_planners.py

Tests for APF, RRT* and the hybrid.

The maps here are built by hand rather than generated, so each test states one behaviour the project claims and checks it on a map where the right answer is obvious: APF is caught by a pocket, RRT* finds its way around a wall, and the hybrid leaves the pocket APF cannot.

The collision tests matter most. A planner that reports success while its route passes through a building would flatter itself in every table in the results.
"""

import unittest

import numpy as np

from planning.algorithms.apf import APF
from planning.algorithms.hybrid import Hybrid
from planning.algorithms.rrt_star import RRTStar
from planning.algorithms.smoothers import BezierSmoother, NoSmoother
from planning.utils.collision import validate_path
from planning.utils.config import Config
from planning.utils.metrics import reached_goal


def test_config(**overrides):
    """A configuration with small iteration budgets so the tests finish quickly."""
    settings = dict(
        grid_size=30, apf_max_iter=1200, rrt_max_iter=2500, out_dir="/tmp/test_outputs"
    )
    settings.update(overrides)
    return Config(**settings)


def open_map(size=30):
    """A square map with nothing on it."""
    return np.zeros((size, size), dtype=np.uint8)


def wall_with_a_gap(size=30):
    """A map split by a wall that has one gap near the bottom, so the only route is through the gap."""
    grid = open_map(size)
    grid[:, size // 2] = 1
    grid[2:6, size // 2] = 0
    return grid


def pocket_map(size=30):
    """
    A map with a three sided pocket whose opening faces the start. The attractive pull towards the goal points straight into the pocket, which is the local minimum APF cannot leave on its own.
    """
    grid = open_map(size)

    # The back wall of the pocket, between the vehicle and the goal
    grid[8:23, 18] = 1

    # The two arms reaching back towards the start, which close the pocket on the other two sides
    grid[8, 8:19] = 1
    grid[22, 8:19] = 1

    return grid


class TestAPF(unittest.TestCase):
    def setUp(self):
        self.config = test_config()
        self.rng = np.random.default_rng(1)

    def test_it_crosses_an_open_map(self):
        grid = open_map()
        start, goal = (2, 2), (27, 27)

        result = APF(grid, start, goal, self.config, self.rng).plan()

        self.assertTrue(result["success"])
        self.assertTrue(reached_goal(result["path"], goal, self.config.goal_tolerance))

    def test_its_route_never_passes_through_an_obstacle(self):
        grid = wall_with_a_gap()
        result = APF(grid, (2, 2), (27, 2), self.config, self.rng).plan()

        # Whether it arrives or not, every step it took has to have been driveable
        self.assertTrue(validate_path(grid, result["path"]))

    def test_it_starts_where_it_was_told_to(self):
        result = APF(open_map(), (2, 2), (27, 27), self.config, self.rng).plan()
        np.testing.assert_allclose(result["path"][0], [2, 2])

    def test_the_pocket_traps_it(self):
        grid = pocket_map()

        result = APF(grid, (3, 15), (27, 15), self.config, self.rng).plan()

        # This failure is the reason the hybrid exists. If APF started solving this map the comparison would have nothing to show
        self.assertFalse(result["success"])

    def test_it_never_reports_switching_modes(self):
        result = APF(open_map(), (2, 2), (27, 27), self.config, self.rng).plan()

        # APF has no escape mode, so the column has to be zero rather than missing
        self.assertEqual(result["switches"], 0)


class TestRRTStar(unittest.TestCase):
    def setUp(self):
        self.config = test_config()

    def test_it_crosses_an_open_map(self):
        grid = open_map()
        goal = (27, 27)

        result = RRTStar(grid, (2, 2), goal, self.config, np.random.default_rng(1)).plan()

        self.assertTrue(result["success"])
        self.assertTrue(reached_goal(result["path"], goal, self.config.goal_tolerance))

    def test_it_finds_the_gap_in_the_wall(self):
        grid = wall_with_a_gap()

        result = RRTStar(grid, (2, 20), (27, 20), self.config, np.random.default_rng(2)).plan()

        self.assertTrue(result["success"])

    def test_its_branches_never_pass_through_an_obstacle(self):
        # This is the regression test for the tree growing straight through buildings, which it did while the branch from the nearest node went unchecked
        grid = wall_with_a_gap()

        for seed in range(6):
            result = RRTStar(
                grid, (2, 20), (27, 20), self.config, np.random.default_rng(seed)
            ).plan()

            if result["success"]:
                self.assertTrue(
                    validate_path(grid, result["path"]),
                    f"seed {seed} produced a route that is not driveable",
                )

    def test_it_leaves_the_pocket_alone(self):
        # A pocket is no obstacle to a sampling planner, which is the other half of the reason the two are combined
        grid = pocket_map()

        result = RRTStar(grid, (3, 15), (27, 15), self.config, np.random.default_rng(3)).plan()

        self.assertTrue(result["success"])
        self.assertTrue(validate_path(grid, result["path"]))

    def test_it_reports_failure_when_the_goal_is_walled_off(self):
        grid = open_map()
        grid[:, 15] = 1

        result = RRTStar(
            grid, (2, 2), (27, 2), test_config(rrt_max_iter=400), np.random.default_rng(4)
        ).plan()

        # A planner that cannot arrive has to say so rather than return a route that goes through the wall
        self.assertFalse(result["success"])
        self.assertIsNone(result["path"])

    def test_the_same_seed_gives_the_same_tree(self):
        grid = wall_with_a_gap()

        first = RRTStar(grid, (2, 20), (27, 20), self.config, np.random.default_rng(7)).plan()
        second = RRTStar(grid, (2, 20), (27, 20), self.config, np.random.default_rng(7)).plan()

        # Every stochastic run in the experiment is seeded, so this is what makes the recorded results reproducible
        np.testing.assert_allclose(np.asarray(first["path"]), np.asarray(second["path"]))


class TestHybrid(unittest.TestCase):
    def setUp(self):
        self.config = test_config()

    def build(self, grid, start, goal, adaptive=False, smoother=None, seed=1):
        smoother = smoother if smoother is not None else NoSmoother(self.config)
        return Hybrid(
            grid, start, goal, self.config, np.random.default_rng(seed),
            adaptive=adaptive, smoother=smoother,
        )

    def test_it_escapes_the_pocket_that_traps_apf(self):
        grid = pocket_map()
        start, goal = (3, 15), (27, 15)

        trapped = APF(grid, start, goal, self.config, np.random.default_rng(1)).plan()
        escaped = self.build(grid, start, goal, adaptive=True).plan()

        # The one claim the whole project rests on
        self.assertFalse(trapped["success"])
        self.assertTrue(escaped["success"])
        self.assertGreater(escaped["switches"], 0)

    def test_its_route_is_driveable_after_an_escape(self):
        grid = pocket_map()

        result = self.build(grid, (3, 15), (27, 15), adaptive=True).plan()

        # The escape is stitched into the middle of an APF route, so the join is where a broken route would show up
        self.assertTrue(validate_path(grid, result["path"]))

    def test_it_records_where_the_escapes_happened(self):
        grid = pocket_map()

        result = self.build(grid, (3, 15), (27, 15), adaptive=True).plan()

        # The junctions drive the markers on the figures, so they have to index the path that was returned
        self.assertEqual(len(result["junctions"]), result["switches"])
        for index in result["junctions"]:
            self.assertTrue(0 <= index < len(result["path"]))

    def test_it_behaves_like_apf_where_there_is_nothing_to_escape(self):
        grid = open_map()

        hybrid = self.build(grid, (2, 2), (27, 27)).plan()
        apf = APF(grid, (2, 2), (27, 27), self.config, np.random.default_rng(1)).plan()

        # With no trap to leave, the hybrid should never pay for an escape it did not need
        self.assertEqual(hybrid["switches"], 0)
        np.testing.assert_allclose(
            np.asarray(hybrid["path"]), np.asarray(apf["path"])
        )

    def test_smoothing_leaves_the_two_ends_where_they_were(self):
        grid = pocket_map()
        start, goal = (3, 15), (27, 15)

        smoothed = self.build(
            grid, start, goal, adaptive=True, smoother=BezierSmoother(self.config)
        ).plan()

        # A smoother that moved the last point would break arrival, and one that moved the first would start the vehicle somewhere it never was
        np.testing.assert_allclose(smoothed["path"][0], start)
        self.assertTrue(reached_goal(smoothed["path"], goal, self.config.goal_tolerance))

    def test_smoothing_keeps_the_route_driveable(self):
        grid = pocket_map()

        smoothed = self.build(
            grid, (3, 15), (27, 15), adaptive=True, smoother=BezierSmoother(self.config)
        ).plan()

        # The smoother is allowed to cut a corner, but never through a building
        self.assertTrue(validate_path(grid, smoothed["path"]))

    def test_the_same_seed_gives_the_same_run(self):
        grid = pocket_map()

        first = self.build(grid, (3, 15), (27, 15), adaptive=True, seed=5).plan()
        second = self.build(grid, (3, 15), (27, 15), adaptive=True, seed=5).plan()

        np.testing.assert_allclose(np.asarray(first["path"]), np.asarray(second["path"]))


class TestAdaptiveStuckDetection(unittest.TestCase):
    def setUp(self):
        self.config = test_config()
        self.hybrid = Hybrid(
            open_map(), (2, 2), (27, 27), self.config, np.random.default_rng(1), adaptive=True
        )

    def test_open_space_gives_the_largest_window(self):
        # With nothing around, the planner should be patient rather than declaring itself stuck the moment progress slows
        self.assertEqual(self.hybrid._adaptive_window(0.0), self.config.stuck_window_max)

    def test_a_packed_area_gives_the_smallest_window(self):
        # Surrounded by obstacles, waiting only wastes iterations
        self.assertEqual(self.hybrid._adaptive_window(1.0), self.config.stuck_window_min)

    def test_the_window_falls_as_the_area_gets_more_crowded(self):
        windows = [self.hybrid._adaptive_window(d) for d in (0.0, 0.25, 0.5, 0.75, 1.0)]

        for earlier, later in zip(windows, windows[1:]):
            self.assertGreaterEqual(earlier, later)

    def test_the_window_never_leaves_the_configured_range(self):
        for density in np.linspace(0.0, 1.0, 21):
            window = self.hybrid._adaptive_window(float(density))
            self.assertTrue(
                self.config.stuck_window_min <= window <= self.config.stuck_window_max
            )


class TestLocalDensity(unittest.TestCase):
    def setUp(self):
        self.config = test_config()

    def density(self, grid, x, y, radius=4):
        hybrid = Hybrid(
            grid, (1, 1), (5, 5), self.config, np.random.default_rng(1), adaptive=True
        )
        return hybrid._local_density(x, y, radius)

    def test_open_ground_reads_as_empty(self):
        self.assertAlmostEqual(self.density(open_map(), 15, 15), 0.0)

    def test_solid_ground_reads_as_full(self):
        grid = np.ones((30, 30), dtype=np.uint8)
        self.assertAlmostEqual(self.density(grid, 15, 15), 1.0)

    def test_a_corner_of_the_map_reads_as_crowded(self):
        # Cells off the edge count as blocked, so the planner treats a boundary wall the way it treats a building
        self.assertGreater(self.density(open_map(), 0, 0), 0.5)

    def test_rubble_counts_the_same_as_a_building(self):
        building = open_map()
        building[10:20, 10:20] = 1

        rubble = open_map()
        rubble[10:20, 10:20] = 2

        self.assertAlmostEqual(self.density(building, 15, 15), self.density(rubble, 15, 15))


if __name__ == "__main__":
    unittest.main()


class TestAStepCannotClipACorner(unittest.TestCase):
    """
    The scorer checks every segment of a route, sampled every half cell. A planner that only checks the cell its step lands on can pass its own test and fail the scorer's on the same route, by cutting diagonally through the corner of an obstacle cell. The planners have to check the step the same way the scorer does.
    """

    def setUp(self):
        # Repulsion is switched off so the vehicle drives a straight line and the geometry of the clip is fixed. The start and goal are chosen so the line crosses cell (5, 4) between two steps without either step landing on it
        self.config = test_config(apf_k_rep=0.0, max_escapes=0)
        self.grid = open_map(12)
        self.grid[4, 5] = 1
        self.start = np.array([2.0, 2.6])
        self.goal = np.array([8.0, 7.0])

    def check(self, result):
        # Driving through the corner is not a success, and the route driven up to the block has to be clean
        self.assertFalse(result["success"])
        self.assertTrue(validate_path(self.grid, result["path"]))

    def test_apf(self):
        self.check(
            APF(self.grid, self.start, self.goal, self.config, np.random.default_rng(0)).plan()
        )

    def test_hybrid(self):
        self.check(
            Hybrid(
                self.grid, self.start, self.goal, self.config, np.random.default_rng(0),
                adaptive=False, smoother=NoSmoother(self.config),
            ).plan()
        )


class TestAWallTouchWaitsForTheWindow(unittest.TestCase):
    """
    A step that would drive into an obstacle is refused, and the vehicle stays put. That on its own is not proof it is stuck: the window has to see no progress for the usual number of steps before an escape is spent on it. Otherwise one brush with a wall costs a full RRT* run and counts as a switch.
    """

    def test_the_hybrid_keeps_trying_for_a_window_before_it_switches(self):
        # No repulsion, so the vehicle drives straight at the wall and is refused there. No escapes allowed, so the run ends at the first stuck call and iters says when that was
        config = test_config(apf_k_rep=0.0, max_escapes=0, stuck_window_fixed=12)
        grid = open_map(30)
        grid[:, 15] = 1

        result = Hybrid(
            grid, (5.0, 15.0), (25.0, 15.0), config, np.random.default_rng(0),
            adaptive=False, smoother=NoSmoother(config),
        ).plan()

        # One iteration per step driven, then one refused step per iteration until the window sees no displacement. The last few driven steps already fall under stuck_delta, so the window fills that many refused steps sooner than its full length
        steps_driven = len(result["path"]) - 1
        steps_under_delta = int(np.ceil(config.stuck_delta / config.apf_step))
        self.assertFalse(result["success"])
        self.assertGreaterEqual(
            result["iters"], steps_driven + config.stuck_window_fixed - 1 - steps_under_delta
        )


class TestSubGoalLooksAhead(unittest.TestCase):
    """
    The escape aims a fixed distance along the line to the goal. When that point sits in a pocket the escape hands the vehicle straight back to the trap, so the sub-goal has to look one more sub-goal distance ahead and aim further when that line is blocked.
    """

    def setUp(self):
        self.config = test_config(subgoal_dist=10.0)

    def hybrid(self, grid, start, goal):
        return Hybrid(grid, start, goal, self.config, np.random.default_rng(0), adaptive=False, smoother=NoSmoother(self.config))

    def test_on_an_open_map_the_sub_goal_sits_at_the_configured_distance(self):
        planner = self.hybrid(open_map(40), (5.0, 20.0), (35.0, 20.0))

        subgoal = planner._pick_subgoal(np.array([5.0, 20.0]))
        self.assertAlmostEqual(float(np.linalg.norm(subgoal - np.array([5.0, 20.0]))), 10.0)

    def test_a_sub_goal_in_a_pocket_is_pushed_past_it(self):
        # A U open towards the start, sitting so the default sub-goal lands inside it and the back wall blocks the line onwards
        grid = open_map(40)
        grid[14:27, 20] = 1
        grid[14, 8:21] = 1
        grid[26, 8:21] = 1
        planner = self.hybrid(grid, (5.0, 20.0), (35.0, 20.0))

        subgoal = planner._pick_subgoal(np.array([5.0, 20.0]))

        # Past the back wall, on the far side of the U
        self.assertGreater(subgoal[0], 20.0)

    def test_the_doublings_are_capped(self):
        # A wall right across the map with no way past it in a straight line. The sub-goal still comes back rather than looping, and no further than the cap allows
        grid = open_map(40)
        grid[:, 30] = 1
        planner = self.hybrid(grid, (5.0, 20.0), (35.0, 20.0))

        subgoal = planner._pick_subgoal(np.array([5.0, 20.0]))
        furthest = self.config.subgoal_dist * 2 ** self.config.subgoal_max_doublings
        self.assertLessEqual(float(np.linalg.norm(subgoal - np.array([5.0, 20.0]))), furthest + 1.0)


class TestJunctionsSurviveSmoothing(unittest.TestCase):
    """
    The hybrid reports where it switched to RRT* as indices into its path. Smoothing replaces stretches of that path with curves of a different point count, so an index into the smoothed path points at the wrong place. The switch indices have to index the route as it was driven, which the result hands back as raw_path.
    """

    def run_hybrid(self, smoother):
        config = test_config()
        grid = open_map(40)
        # A pocket across the direct route so the hybrid has to escape at least once
        grid[10:30, 25] = 1
        grid[10, 12:26] = 1
        grid[29, 12:26] = 1
        return Hybrid(grid, (3.0, 20.0), (36.0, 20.0), config, np.random.default_rng(0), adaptive=False, smoother=smoother).plan()

    def test_the_raw_path_is_the_route_before_smoothing(self):
        config = test_config()
        plain = self.run_hybrid(NoSmoother(config))
        smoothed = self.run_hybrid(BezierSmoother(config))

        self.assertGreater(len(plain["junctions"]), 0)
        np.testing.assert_allclose(np.asarray(smoothed["raw_path"]), np.asarray(plain["path"]))
        self.assertEqual(smoothed["junctions"], plain["junctions"])

    def test_every_junction_indexes_the_raw_path(self):
        result = self.run_hybrid(BezierSmoother(test_config()))
        for junction in result["junctions"]:
            self.assertLess(junction, len(result["raw_path"]))


class TestPlannerTrace(unittest.TestCase):
    """
    The animation needs to see the RRT* tree grow and the hybrid hand over. The planners record that when asked, and only when asked, so the benchmark's timing and its random draws are untouched.
    """

    def setUp(self):
        self.config = test_config()
        self.grid = wall_with_a_gap()

    def test_the_trace_changes_nothing_about_the_route(self):
        plain = RRTStar(self.grid, (3.0, 3.0), (26.0, 26.0), self.config, np.random.default_rng(4)).plan()
        traced = RRTStar(self.grid, (3.0, 3.0), (26.0, 26.0), self.config, np.random.default_rng(4), trace=True).plan()

        np.testing.assert_array_equal(np.asarray(plain["path"]), np.asarray(traced["path"]))
        self.assertNotIn("trace", plain)
        self.assertGreater(len(traced["trace"]), 0)

    def test_every_added_node_hangs_off_a_node_already_in_the_tree(self):
        result = RRTStar(self.grid, (3.0, 3.0), (26.0, 26.0), self.config, np.random.default_rng(4), trace=True).plan()

        known = {(3.0, 3.0)}
        rewires = 0
        for event in result["trace"]:
            if event[0] == "add":
                _, node, parent = event
                self.assertIn(tuple(parent), known)
                known.add(tuple(node))
            elif event[0] == "rewire":
                _, node, old_parent, new_parent = event
                self.assertIn(tuple(node), known)
                self.assertIn(tuple(new_parent), known)
                self.assertNotEqual(tuple(old_parent), tuple(new_parent))
                rewires += 1
            else:
                self.assertEqual(event[0], "goal")

        # A tree that never rewired would not be RRT*
        self.assertGreater(rewires, 0)
        self.assertEqual(result["trace"][-1][0], "goal")

    def test_the_hybrid_records_one_escape_trace_per_switch(self):
        grid = open_map(40)
        grid[10:30, 25] = 1
        grid[10, 12:26] = 1
        grid[29, 12:26] = 1

        plain = Hybrid(grid, (3.0, 20.0), (36.0, 20.0), self.config, np.random.default_rng(0), adaptive=False, smoother=NoSmoother(self.config)).plan()
        traced = Hybrid(grid, (3.0, 20.0), (36.0, 20.0), self.config, np.random.default_rng(0), adaptive=False, smoother=NoSmoother(self.config), trace=True).plan()

        np.testing.assert_array_equal(np.asarray(plain["path"]), np.asarray(traced["path"]))
        self.assertGreater(traced["switches"], 0)
        self.assertEqual(len(traced["trace"]), traced["switches"])

        # Each escape trace says where in the driven route the switch happened and holds the tree that grew there
        for escape, junction in zip(traced["trace"], traced["junctions"]):
            self.assertEqual(escape["junction"], junction)
            self.assertGreater(len(escape["events"]), 0)
