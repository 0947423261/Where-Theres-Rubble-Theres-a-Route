"""
tests/test_determinism.py

The same seed has to give the same path, every time.

Every number in the results chapter rests on this. If a planner reads the global generator, or consumes its own generator in an order that depends on anything but the seed, then a rerun of one recorded row gives a different path and the whole table is unrepeatable. These tests build the maps and seed the runs the same way the harness does, so they check the experiment as it is actually run rather than a hand-made grid.
"""

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from planning.experiments.runner import DynamicStudy, build_algorithms, derived_seed
from planning.maps.moving_obstacle import make_dynamic_instance
from planning.maps.scenarios import make_instance
from planning.utils.config import Config


def small_config():
    """A configuration small enough that four planners on two maps finish inside a test."""
    return Config(
        grid_size=30,
        start=(2, 3),
        goal=(27, 26),
        apf_max_iter=600,
        rrt_max_iter=800,
        out_dir=tempfile.mkdtemp(),
    )


SCENARIOS = ("normal_city", "blocked_road")
SPECS = {spec.name: spec for spec in build_algorithms()}
DRIVERS = [name for name, _ in DynamicStudy(small_config()).build_drivers()]


def run_once(spec, scenario, config, instance=0):
    """Run one planner on one map exactly as the harness does and return its result."""
    grid, start, goal = make_instance(scenario, config.master_seed + instance, config)

    # A fresh generator per run, seeded from the same parts the runner uses
    rng = np.random.default_rng(derived_seed(config.master_seed, scenario, instance, spec.seed_name))
    return spec.plan(grid, start, goal, config, rng)


def run_driver_once(name, config, instance=0):
    """Run one dynamic driver on one corridor exactly as DynamicStudy does and return its result."""
    grid, start, goal, obstacle = make_dynamic_instance(config.master_seed + instance, config)
    factory = dict(DynamicStudy(config).build_drivers())[name]

    rng = np.random.default_rng(derived_seed(config.master_seed, "dynamic", instance, name))
    return factory(grid, start, goal, rng, obstacle).plan()


def path_array(result):
    return np.asarray(result["path"], dtype=float)


class TestSamePathEveryRun(unittest.TestCase):
    """Three consecutive runs from the same seed give byte-identical paths. Exact equality, not a tolerance, since a float that drifts is a float that was computed from something other than the seed."""

    def setUp(self):
        self.config = small_config()

    def check(self, name):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                results = [run_once(SPECS[name], scenario, self.config) for _ in range(3)]
                first = path_array(results[0])

                for later in results[1:]:
                    self.assertTrue(np.array_equal(first, path_array(later)))
                    self.assertEqual(results[0]["success"], later["success"])
                    self.assertEqual(results[0]["switches"], later["switches"])

    def test_apf(self):
        self.check("APF")

    def test_rrt_star(self):
        self.check("RRT*")

    def test_rrt_star_with_bezier(self):
        self.check("RRT*+Bezier")

    def test_hybrid(self):
        self.check("Hybrid")

    def test_modified_hybrid(self):
        self.check("Modified")


class TestSameDriveEveryRun(unittest.TestCase):
    """The moving obstacle study has more places to go wrong: the obstacle walks on its own generator, the drivers replan mid-route, and the obstacle is shared between drivers. Same check, byte-identical trace over three runs."""

    def setUp(self):
        self.config = small_config()

    def check(self, name):
        results = [run_driver_once(name, self.config) for _ in range(3)]
        first = path_array(results[0])

        for later in results[1:]:
            self.assertTrue(np.array_equal(first, path_array(later)))
            for key in ("success", "collided", "replans", "iters"):
                self.assertEqual(results[0][key], later[key])

    def test_apf_reactive(self):
        self.check("APF reactive")

    def test_rrt_star_plan_once(self):
        self.check("RRT* plan once")

    def test_rrt_star_replan(self):
        self.check("RRT* replan")

    def test_modified_plan_once(self):
        self.check("Modified plan once")

    def test_modified_replan(self):
        self.check("Modified replan")

    def test_the_obstacle_walks_the_same_way_whoever_asks(self):
        # The obstacle is shared between the drivers in the study, so the walk must not depend on which driver asked for which step first
        _, _, _, alone = make_dynamic_instance(42, self.config)
        _, _, _, shared = make_dynamic_instance(42, self.config)
        for name in DRIVERS:
            run_driver_once(name, self.config)

        for t in range(0, 40, 7):
            self.assertTrue(np.array_equal(alone.position_at(t), shared.position_at(t)))


class TestNothingReadsTheGlobalGenerator(unittest.TestCase):
    def test_a_run_leaves_the_global_stream_untouched(self):
        config = small_config()

        # If any planner or driver drew from np.random instead of its own generator, the global stream would sit further along after the run
        np.random.seed(0)
        expected = np.random.random()

        np.random.seed(0)
        for name in SPECS:
            run_once(SPECS[name], "blocked_road", config)
        for name in DRIVERS:
            run_driver_once(name, config)
        self.assertEqual(np.random.random(), expected)


class TestTheMapIsReproducible(unittest.TestCase):
    def test_the_same_seed_rebuilds_the_same_map(self):
        config = small_config()
        for scenario in SCENARIOS:
            grid_a, _, _ = make_instance(scenario, 42, config)
            grid_b, _, _ = make_instance(scenario, 42, config)
            self.assertTrue(np.array_equal(grid_a, grid_b))


class TestAFreshInterpreterAgrees(unittest.TestCase):
    """The in-process tests cannot see state that is fixed for the life of one process, such as hash randomisation, so one planner is also run in three separate interpreters."""

    SCRIPT = """
import hashlib, sys
sys.path.insert(0, {root!r})
from tests.test_determinism import SPECS, run_once, small_config, path_array
result = run_once(SPECS["Modified"], "blocked_road", small_config())
print(hashlib.sha256(path_array(result).tobytes()).hexdigest())
"""

    def test_three_processes_give_the_same_path(self):
        root = str(Path(__file__).resolve().parent.parent)
        digests = set()

        for _ in range(3):
            output = subprocess.run(
                [sys.executable, "-c", self.SCRIPT.format(root=root)],
                capture_output=True, text=True, check=True, cwd=root,
            )
            digests.add(output.stdout.strip())

        self.assertEqual(len(digests), 1)


if __name__ == "__main__":
    unittest.main()
