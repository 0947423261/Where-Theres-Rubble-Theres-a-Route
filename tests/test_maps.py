"""
tests/test_maps.py

Tests for the map generators and the moving obstacle.

The point of most of these is reproducibility. The whole experiment rests on the claim that instance i of a scenario is the same map for every algorithm and on every day the code is run, so that claim is worth testing rather than assuming.
"""

import unittest

import numpy as np

from planning.maps.moving_obstacle import (
    MOVING_VALUE,
    MovingObstacle,
    make_dynamic_instance,
)
from planning.maps.inflate import MARGIN_VALUE, inflate_obstacles
from planning.maps.scenarios import SCENARIOS, Scenario, make_instance
from planning.utils.collision import point_blocked, validate_path
from planning.utils.config import Config


def small_config(**overrides):
    """A small configuration so the tests build quickly."""
    settings = dict(grid_size=40, start=(2, 3), goal=(37, 36), out_dir="/tmp/test_outputs")
    settings.update(overrides)
    return Config(**settings)


class TestScenarioGeneration(unittest.TestCase):
    def setUp(self):
        self.config = small_config()

    def test_every_registered_scenario_builds(self):
        for name in SCENARIOS:
            grid, start, goal = make_instance(name, 7, self.config)

            # The map is square and the size the configuration asked for
            self.assertEqual(grid.shape, (self.config.grid_size, self.config.grid_size))

            # Both ends are on the map and free to stand on
            self.assertFalse(point_blocked(grid, start[0], start[1]))
            self.assertFalse(point_blocked(grid, goal[0], goal[1]))

    def test_only_the_documented_cell_values_appear(self):
        for name in SCENARIOS:
            grid, _, _ = make_instance(name, 3, self.config)

            # Streets, buildings, rubble and the inflation margin. The moving obstacle is never baked into a static map
            self.assertTrue(set(np.unique(grid)).issubset({0, 1, 2, MARGIN_VALUE}))

    def test_the_same_seed_rebuilds_the_same_map(self):
        first, _, _ = make_instance("dense_city", 11, self.config)
        second, _, _ = make_instance("dense_city", 11, self.config)

        # This is what makes the comparison paired: every algorithm has to meet the same map
        np.testing.assert_array_equal(first, second)

    def test_different_seeds_give_different_maps(self):
        first, _, _ = make_instance("dense_city", 11, self.config)
        second, _, _ = make_instance("dense_city", 12, self.config)

        # Without this the fifty instances would be fifty copies of one map
        self.assertFalse(np.array_equal(first, second))

    def test_every_generated_map_is_solvable(self):
        for name in SCENARIOS:
            for seed in range(5):
                grid, start, goal = make_instance(name, seed, self.config)

                # A failure in the results has to be the fault of the algorithm and never of an impossible map
                self.assertTrue(Scenario._reachable(grid, start, goal))

    def test_every_obstacle_carries_a_margin_of_the_vehicle_radius(self):
        # With a radius of one cell every cell touching an obstacle, including diagonally, has to be blocked on the map the planners see
        for name in SCENARIOS:
            grid, _, _ = make_instance(name, 4, self.config)
            real = (grid == 1) | (grid == 2)
            h, w = grid.shape

            for y in range(h):
                for x in range(w):
                    if not real[y, x]:
                        continue
                    for change_y in (-1, 0, 1):
                        for change_x in (-1, 0, 1):
                            nx, ny = x + change_x, y + change_y
                            if 0 <= nx < w and 0 <= ny < h:
                                self.assertNotEqual(grid[ny, nx], 0, (name, nx, ny))

    def test_a_zero_radius_adds_no_margin(self):
        grid, _, _ = make_instance("normal_city", 4, small_config(vehicle_radius=0.0))
        self.assertEqual(np.count_nonzero(grid == MARGIN_VALUE), 0)

    def test_a_street_closure_seals_one_street_crosswise(self):
        # A closure that runs along a street instead of across it grows until it meets a building, which on a continuous street is the far edge of the map. That wall cuts the map in two. A closure has to span one street's width and no more
        grid = np.zeros((40, 40), dtype=np.uint8)
        Scenario._carve_street_grid(grid, np.random.default_rng(0), street_width=5, block_min=7, block_max=10)

        for seed in range(40):
            closed = grid.copy()
            Scenario._close_street(closed, np.random.default_rng(seed))
            wall_cells = int(np.count_nonzero(closed == 2))

            # Across one street the wall is five cells; across an intersection it can reach the far building of the crossing street, so twice the width is the most it can honestly be
            with self.subTest(seed=seed):
                self.assertLessEqual(wall_cells, 2 * 5)

    def test_the_blocked_road_hands_out_its_pocket(self):
        from planning.maps.scenarios import make_scenario

        grid, start, goal, scenario = make_scenario("blocked_road", 4, self.config)
        x_min, x_max, y_min, y_max = scenario.pocket

        # The pocket is a rectangle inside the map with the back wall beyond it on the goal side
        self.assertLess(x_min, x_max)
        self.assertLess(y_min, y_max)
        self.assertTrue(np.any(grid[y_max + 1 : y_max + 3, x_min:x_max] == 2))

    def test_the_other_scenarios_have_no_pocket(self):
        from planning.maps.scenarios import make_scenario

        _, _, _, scenario = make_scenario("normal_city", 4, self.config)
        self.assertIsNone(getattr(scenario, "pocket", None))

    def test_an_unknown_scenario_is_refused(self):
        with self.assertRaises(ValueError):
            make_instance("suburbs", 1, self.config)

    def test_the_dense_city_is_denser_than_the_normal_one(self):
        normal, _, _ = make_instance("normal_city", 5, self.config)
        dense, _, _ = make_instance("dense_city", 5, self.config)

        # The scenario names have to mean what they say, otherwise the difficulty ordering in the results is meaningless
        self.assertGreater(np.count_nonzero(dense), np.count_nonzero(normal))

    def test_the_blocked_road_adds_rubble_to_the_normal_city(self):
        normal, _, _ = make_instance("normal_city", 5, self.config)
        blocked, _, _ = make_instance("blocked_road", 5, self.config)

        # The trap is built out of rubble, so the blocked map has to carry more of it
        self.assertGreater(np.count_nonzero(blocked == 2), np.count_nonzero(normal == 2))


class TestReachability(unittest.TestCase):
    def test_a_walled_off_goal_is_unreachable(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[:, 5] = 1

        self.assertFalse(Scenario._reachable(grid, (1, 1), (9, 1)))

    def test_a_gap_in_the_wall_makes_it_reachable(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[:, 5] = 1
        grid[8, 5] = 0

        self.assertTrue(Scenario._reachable(grid, (1, 1), (9, 1)))

    def test_a_buried_end_is_unreachable(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[1, 1] = 1

        self.assertFalse(Scenario._reachable(grid, (1, 1), (8, 8)))


class TestMovingObstacle(unittest.TestCase):
    def build(self, seed=1):
        return MovingObstacle(
            (40, 40), seed=seed, half_size=3, speed=1.0, start=(20, 20)
        )

    def test_the_walk_is_the_same_every_time_it_is_asked_for(self):
        obstacle = self.build()

        # Asking twice has to give the same answer, or the vehicle and the animation would disagree about where the obstacle was
        first = obstacle.position_at(15).copy()
        second = obstacle.position_at(15).copy()
        np.testing.assert_array_equal(first, second)

    def test_asking_out_of_order_gives_the_same_walk(self):
        forwards = self.build()
        for t in range(20):
            forwards.position_at(t)

        # Jumping straight to a later step has to walk the same route as stepping through it
        jumped = self.build()
        np.testing.assert_allclose(jumped.position_at(19), forwards.position_at(19))

    def test_the_same_seed_gives_the_same_walk(self):
        np.testing.assert_allclose(
            self.build(seed=4).position_at(30), self.build(seed=4).position_at(30)
        )

    def test_different_seeds_give_different_walks(self):
        self.assertFalse(
            np.allclose(self.build(seed=4).position_at(30), self.build(seed=5).position_at(30))
        )

    def test_the_obstacle_stays_inside_its_box(self):
        obstacle = MovingObstacle(
            (40, 40), seed=2, half_size=3, speed=1.4, start=(20, 20), bounds=(15, 25, 10, 30)
        )

        for t in range(300):
            x, y = obstacle.position_at(t)

            # The bounce has to hold across a long walk, not only on the first few steps
            self.assertTrue(15 <= x <= 25)
            self.assertTrue(10 <= y <= 30)

    def test_it_moves_at_the_speed_it_was_given(self):
        obstacle = self.build()

        for t in range(1, 20):
            hop = np.linalg.norm(obstacle.position_at(t) - obstacle.position_at(t - 1))

            # A bounce shortens a step, so the speed is an upper bound rather than an exact value
            self.assertLessEqual(hop, 1.0 + 1e-6)

    def test_stamping_leaves_the_static_map_alone(self):
        obstacle = self.build()
        base = np.zeros((40, 40), dtype=np.uint8)

        stamped = obstacle.stamp(base, 5)

        # The static map is reused on every step, so stamping has to work on a copy
        self.assertEqual(np.count_nonzero(base), 0)
        self.assertGreater(np.count_nonzero(stamped == MOVING_VALUE), 0)

    def test_stamping_and_the_overlap_test_agree_on_every_cell(self):
        # A replanned route is checked against the stamped grid and then driven against the overlap test. If the two disagree by even one cell, a route that passed planning is hit on the next step
        obstacle = self.build()
        base = np.zeros((40, 40), dtype=np.uint8)

        for vehicle_radius in (0.0, 1.0):
            for t in (0, 5, 17):
                stamped = obstacle.stamp(base, t, vehicle_radius)

                for y in range(40):
                    for x in range(40):
                        with self.subTest(t=t, vehicle_radius=vehicle_radius, x=x, y=y):
                            self.assertEqual(
                                stamped[y, x] == MOVING_VALUE,
                                obstacle.hits(x, y, t, vehicle_radius),
                            )

    def test_the_overlap_test_agrees_with_where_the_obstacle_is(self):
        obstacle = self.build()
        x, y = obstacle.position_at(0)

        # Standing on the obstacle is a hit and standing well clear of it is not
        self.assertTrue(obstacle.hits(x, y, 0, vehicle_radius=1.0))
        self.assertFalse(obstacle.hits(x + 20, y, 0, vehicle_radius=1.0))


class TestDynamicCorridor(unittest.TestCase):
    def test_the_corridor_is_open_from_end_to_end(self):
        config = small_config()
        grid, start, goal, obstacle = make_dynamic_instance(config.master_seed, config)

        # Both ends are free to stand on
        self.assertFalse(point_blocked(grid, start[0], start[1]))
        self.assertFalse(point_blocked(grid, goal[0], goal[1]))

        # The corridor is what makes the scenario a fair test, so it has to be clear before the obstacle starts moving
        self.assertTrue(Scenario._reachable(grid, start, goal))

    def test_the_obstacle_patrols_inside_the_corridor(self):
        config = small_config()
        grid, _, _, obstacle = make_dynamic_instance(config.master_seed, config)
        h, w = grid.shape

        for t in range(200):
            x, y = obstacle.position_at(t)

            # An obstacle that walked into a wall would block nothing and the scenario would measure nothing
            self.assertTrue(0 <= x < w and 0 <= y < h)

    def test_the_same_seed_rebuilds_the_same_scenario(self):
        config = small_config()
        first, _, _, first_obstacle = make_dynamic_instance(9, config)
        second, _, _, second_obstacle = make_dynamic_instance(9, config)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(
            first_obstacle.position_at(25), second_obstacle.position_at(25)
        )


if __name__ == "__main__":
    unittest.main()


class TestObstacleInflation(unittest.TestCase):
    """
    The vehicle is not a point. A route that clears the drawn obstacles by less than the vehicle radius would put the vehicle's body through the wall, so it has to score as a collision on the inflated map.
    """

    def wall_map(self, vehicle_radius):
        grid = np.zeros((20, 20), dtype=np.uint8)
        grid[:, 10] = 1
        inflate_obstacles(grid, vehicle_radius)
        return grid

    def test_a_route_hugging_a_wall_is_a_collision(self):
        grid = self.wall_map(vehicle_radius=1.0)

        # Straight up the column next to the wall, never touching the wall itself
        hugging = [np.array([9.0, y]) for y in range(2, 18)]
        self.assertFalse(validate_path(grid, hugging))

    def test_a_route_one_radius_clear_of_the_wall_is_fine(self):
        grid = self.wall_map(vehicle_radius=1.0)

        clear = [np.array([8.0, y]) for y in range(2, 18)]
        self.assertTrue(validate_path(grid, clear))

    def test_the_margin_is_marked_with_its_own_value(self):
        grid = self.wall_map(vehicle_radius=1.0)

        # The wall keeps its value and the margin gets its own, so a drawing can tell them apart
        self.assertTrue(np.all(grid[:, 10] == 1))
        self.assertTrue(np.all(grid[:, 9] == MARGIN_VALUE))
        self.assertTrue(np.all(grid[:, 11] == MARGIN_VALUE))
        self.assertTrue(np.all(grid[:, 8] == 0))

    def test_a_diagonal_neighbour_is_inside_the_margin(self):
        # A vehicle of radius one at a diagonal offset is 0.707 cells from the corner of the obstacle cell, so it overlaps
        grid = np.zeros((7, 7), dtype=np.uint8)
        grid[3, 3] = 1
        inflate_obstacles(grid, 1.0)

        self.assertEqual(grid[2, 2], MARGIN_VALUE)
        self.assertEqual(grid[1, 3], 0)

    def test_the_corridor_walls_carry_a_margin_too(self):
        config = small_config()
        grid, _, _, _ = make_dynamic_instance(config.master_seed, config)

        # The corridor walls are solid rows, so the row inside each wall is margin
        wall = max(4, config.grid_size // 2 - config.dynamic_corridor_half)
        self.assertTrue(np.all(grid[wall, :] == MARGIN_VALUE))
        self.assertTrue(np.all(grid[config.grid_size - wall - 1, :] == MARGIN_VALUE))


class TestObstacleSpeedFactor(unittest.TestCase):
    def test_the_obstacle_moves_at_the_factor_times_the_vehicle_speed(self):
        config = small_config()
        _, _, _, obstacle = make_dynamic_instance(config.master_seed, config, speed_factor=0.5)
        self.assertAlmostEqual(obstacle.speed, 0.5 * config.apf_step)

    def test_the_default_factor_is_the_vehicle_speed(self):
        config = small_config()
        _, _, _, obstacle = make_dynamic_instance(config.master_seed, config)
        self.assertAlmostEqual(obstacle.speed, config.apf_step)

    def test_two_factors_on_one_seed_walk_differently_from_the_same_start(self):
        config = small_config()
        _, _, _, slow = make_dynamic_instance(9, config, speed_factor=0.5)
        _, _, _, fast = make_dynamic_instance(9, config, speed_factor=1.5)

        np.testing.assert_allclose(slow.position_at(0), fast.position_at(0))
        self.assertFalse(np.allclose(slow.position_at(10), fast.position_at(10)))

    def test_the_same_factor_and_seed_walk_the_same_way(self):
        config = small_config()
        _, _, _, first = make_dynamic_instance(9, config, speed_factor=1.5)
        _, _, _, second = make_dynamic_instance(9, config, speed_factor=1.5)
        np.testing.assert_allclose(first.position_at(25), second.position_at(25))
