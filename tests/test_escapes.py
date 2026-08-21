"""
tests/test_escapes.py

Where the hybrid's escapes aim.

An escape is meant to get the vehicle out of the pocket that stopped APF. The sub-goal is placed a fixed distance along the straight line to the goal, and on the blocked road that line runs straight through the U. A sub-goal that lands inside the U hands the vehicle straight back to the trap, so the count of those is the measure of whether the escape does its job.
"""

import unittest

import numpy as np

from planning.algorithms.hybrid import Hybrid
from planning.algorithms.smoothers import BezierSmoother, NoSmoother
from planning.experiments.runner import derived_seed
from planning.maps.scenarios import BlockedRoad
from planning.utils.config import Config

# Enough maps to see the pattern, few enough that the hybrid finishes inside a test
INSTANCES = 10


def inside(point, pocket):
    """True when the point lies in the pocket rectangle, inclusive on every side."""
    x_min, x_max, y_min, y_max = pocket
    return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


def escapes_inside_the_trap(name, adaptive, smoother_class, config):
    """
    Run the hybrid the way the harness does on the first few blocked road maps and return (inside, total): how many escapes aimed at a sub-goal inside the U, and how many escapes there were.
    """
    inside_count = 0
    total = 0

    for instance in range(INSTANCES):
        scenario = BlockedRoad(config)
        grid, start, goal = scenario.build(config.master_seed + instance)

        rng = np.random.default_rng(derived_seed(config.master_seed, "blocked_road", instance, name))
        planner = Hybrid(grid, start, goal, config, rng, adaptive=adaptive, smoother=smoother_class(config))
        result = planner.plan()

        # The sub-goal is a pure function of where the vehicle stood when it switched, so it can be asked for again after the run
        for junction in result["junctions"]:
            subgoal = planner._pick_subgoal(result["path"][junction])
            total += 1
            if inside(subgoal, scenario.pocket):
                inside_count += 1

    return inside_count, total


class TestEscapesLeaveTheTrap(unittest.TestCase):
    def setUp(self):
        self.config = Config(out_dir="/tmp/test_outputs")

    def check(self, name, adaptive, smoother_class):
        inside_count, total = escapes_inside_the_trap(name, adaptive, smoother_class, self.config)
        print(f"\n[JR2] {name}: {inside_count} of {total} escapes aimed inside the U over {INSTANCES} maps")

        # The hybrid has to escape on the blocked road, that is what the scenario is for
        self.assertGreater(total, 0)
        self.assertEqual(inside_count, 0)

    # JR2 measured 6 of 24 escapes on the standard hybrid and 7 of 23 on the modified one aiming inside the U before JR3 taught the sub-goal to look ahead
    def test_hybrid(self):
        self.check("Hybrid", adaptive=False, smoother_class=NoSmoother)

    def test_modified(self):
        self.check("Modified", adaptive=True, smoother_class=BezierSmoother)
