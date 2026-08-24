"""
tests/test_golden.py

The regression comparison itself. It has to notice a row that moved and ignore the one column that is allowed to.
"""

import unittest

import pandas as pd

from planning.experiments.golden import STATIC_KEYS, compare, dynamic_path, shared_keys

# The fixture tables predate the repeat column, so they are compared on the keys they have
KEYS = ["scenario", "algorithm", "instance"]


def table(changes=None):
    rows = [
        {"scenario": "normal_city", "algorithm": "APF", "instance": 0, "success": 1, "path_length": 80.123, "comp_time": 0.5, "switches": 0},
        {"scenario": "normal_city", "algorithm": "RRT*", "instance": 0, "success": 1, "path_length": 90.456, "comp_time": 1.5, "switches": 0},
    ]
    for (index, column), value in (changes or {}).items():
        rows[index][column] = value
    return pd.DataFrame(rows)


class TestCompare(unittest.TestCase):
    def test_an_identical_table_has_no_differences(self):
        self.assertEqual(compare(table(), table(), KEYS), [])

    def test_a_moved_value_is_reported_with_both_numbers(self):
        differences = compare(table(), table({(1, "path_length"): 91.0}), KEYS)

        self.assertEqual(len(differences), 1)
        self.assertIn("RRT*", differences[0])
        self.assertIn("90.456", differences[0])
        self.assertIn("91.0", differences[0])

    def test_timing_is_never_compared(self):
        self.assertEqual(compare(table(), table({(0, "comp_time"): 9.9}), KEYS), [])

    def test_a_missing_row_is_a_difference(self):
        self.assertEqual(len(compare(table(), table().iloc[:1], KEYS)), 1)

    def test_a_golden_table_from_before_repeats_is_still_compared(self):
        # The frozen table may predate the repeat column. The keys are the ones both sides carry, so k = 0 rows still line up against it
        old = table()
        fresh = table().assign(repeat=0)
        self.assertEqual(shared_keys(old, fresh, STATIC_KEYS), KEYS)
        self.assertEqual(shared_keys(fresh, fresh, STATIC_KEYS), STATIC_KEYS)

    def test_the_dynamic_table_sits_next_to_the_static_one(self):
        self.assertEqual(dynamic_path("results/golden.csv").name, "golden_dynamic.csv")
