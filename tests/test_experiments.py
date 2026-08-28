"""
tests/test_experiments.py

Tests for the plumbing that turns runs into results.

None of this is an algorithm, and all of it can quietly ruin a results chapter. A seed that changes between sessions makes the experiment unrepeatable, a row whose columns do not match the header writes a file that reads back wrong, and an average that includes the failed runs rewards an algorithm for giving up early.
"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from planning.algorithms.apf import APF
from planning.algorithms.hybrid import Hybrid
from planning.algorithms.rrt_star import RRTStar
from planning.algorithms.smoothers import BezierSmoother, NoSmoother
from planning.experiments.analyse import (
    add_mission_time,
    mask_failures,
    mcnemar,
    paired_tests,
    rank_biserial,
    success_rates,
    wilson_interval,
)
from planning.experiments.record import (
    DYNAMIC_FIELDS,
    STATIC_FIELDS,
    dynamic_row,
    save_rows,
    static_row,
)
from planning.experiments.runner import (
    DynamicStudy,
    PlannerSpec,
    StaticStudy,
    build_algorithms,
    derived_seed,
    run_seed,
)
from planning.maps.scenarios import make_instance
from planning.utils.config import Config


def tiny_config(**overrides):
    """A configuration small enough to run the whole loop inside a test."""
    settings = dict(
        grid_size=30,
        n_instances=2,
        scenarios=("normal_city", "blocked_road"),
        apf_max_iter=600,
        rrt_max_iter=800,
        start=(2, 3),
        goal=(27, 26),
        n_repeats=1,
        out_dir=tempfile.mkdtemp(),
    )
    settings.update(overrides)
    return Config(**settings)


class TestDerivedSeed(unittest.TestCase):
    def test_the_same_inputs_always_give_the_same_seed(self):
        # Python's own hash is randomised per process, so a run seeded with it could never be reproduced on a later day
        self.assertEqual(
            derived_seed(42, "dense_city", 3, "Modified"),
            derived_seed(42, "dense_city", 3, "Modified"),
        )

    def test_different_inputs_give_different_seeds(self):
        seeds = {
            derived_seed(42, "dense_city", i, name)
            for i in range(5)
            for name in ("APF", "RRT*", "Modified")
        }
        self.assertEqual(len(seeds), 15)

    def test_the_parts_cannot_run_into_each_other(self):
        # Without a separator, ("ab", "c") and ("a", "bc") would seed the same run
        self.assertNotEqual(derived_seed("ab", "c"), derived_seed("a", "bc"))

    def test_repeat_zero_is_the_seed_the_old_rows_were_run_with(self):
        # Every row collected before repeats existed was seeded from four parts. Repeat zero has to give that same seed, or the golden table and every recorded run stop being reproducible
        self.assertEqual(run_seed(42, "dense_city", 3, "Modified", 0), derived_seed(42, "dense_city", 3, "Modified"))

    def test_later_repeats_are_different_seeds(self):
        seeds = {run_seed(42, "dense_city", 3, "Modified", k) for k in range(10)}
        self.assertEqual(len(seeds), 10)

    def test_the_seed_fits_what_a_generator_accepts(self):
        for i in range(20):
            seed = derived_seed(42, "scenario", i, "algorithm")
            self.assertTrue(0 <= seed < 2**32)

            # The generator has to accept it without complaint
            np.random.default_rng(seed)


class TestPlannerSpec(unittest.TestCase):
    def setUp(self):
        self.config = tiny_config()
        self.grid = np.zeros((30, 30), dtype=np.uint8)

    def build(self, spec):
        return spec.build(self.grid, (2, 2), (27, 27), self.config, np.random.default_rng(1))

    def test_a_planner_with_no_options_is_built_plainly(self):
        planner = self.build(PlannerSpec("APF", APF))
        self.assertIsInstance(planner, APF)

    def test_the_hybrid_options_are_passed_through(self):
        planner = self.build(
            PlannerSpec("Modified", Hybrid, adaptive=True, smoother_class=BezierSmoother)
        )

        self.assertIsInstance(planner, Hybrid)
        self.assertTrue(planner.adaptive)
        self.assertIsInstance(planner.smoother, BezierSmoother)

    def test_the_standard_hybrid_does_not_smooth(self):
        planner = self.build(
            PlannerSpec("Hybrid", Hybrid, adaptive=False, smoother_class=NoSmoother)
        )

        self.assertFalse(planner.adaptive)
        self.assertIsInstance(planner.smoother, NoSmoother)

    def test_a_fresh_planner_is_built_for_every_run(self):
        spec = PlannerSpec("Modified", Hybrid, adaptive=True, smoother_class=BezierSmoother)

        # The planners hold per-run state, so sharing one between runs would carry a route into the next map
        self.assertIsNot(self.build(spec), self.build(spec))


class TestAlgorithmList(unittest.TestCase):
    def test_the_core_five_are_always_there(self):
        names = [spec.name for spec in build_algorithms()]
        self.assertEqual(names, ["APF", "RRT*", "RRT*+Bezier", "Hybrid", "Modified"])

    def test_rrt_star_with_bezier_smooths_the_same_tree_rrt_star_grew(self):
        # The fifth planner is RRT* with its route smoothed afterwards. It is seeded as RRT* so the two rows differ only by the smoothing, which is what makes the pair comparable
        specs = {spec.name: spec for spec in build_algorithms()}
        self.assertEqual(specs["RRT*+Bezier"].seed_name, "RRT*")
        self.assertEqual(specs["RRT*"].seed_name, "RRT*")

        config = tiny_config()
        grid, start, goal = make_instance("normal_city", config.master_seed, config)
        seed = derived_seed(config.master_seed, "normal_city", 0, "RRT*")

        plain = specs["RRT*"].plan(grid, start, goal, config, np.random.default_rng(seed))
        smoothed = specs["RRT*+Bezier"].plan(grid, start, goal, config, np.random.default_rng(seed))

        expected = BezierSmoother(config).smooth(plain["path"], grid)
        np.testing.assert_allclose(np.asarray(smoothed["path"]), np.asarray(expected))
        self.assertEqual(smoothed["switches"], 0)

    def test_the_ablation_isolates_one_extension_each(self):
        specs = {spec.name: spec for spec in build_algorithms(ablation=True)}

        # Adaptive detection with no smoothing, and smoothing with the fixed window
        self.assertTrue(specs["Adaptive"].adaptive)
        self.assertIs(specs["Adaptive"].smoother_class, NoSmoother)
        self.assertFalse(specs["Bezier"].adaptive)
        self.assertIs(specs["Bezier"].smoother_class, BezierSmoother)

    def test_the_smoother_comparison_holds_everything_else_fixed(self):
        specs = {spec.name: spec for spec in build_algorithms(smoothers=True)}

        # The three smoothing strategies differ only in the strategy, so any difference in the results is the strategy
        for name in ("Modified", "BSpline", "Dubins"):
            self.assertTrue(specs[name].adaptive)
            self.assertIs(specs[name].planner_class, Hybrid)

    def test_every_name_is_unique(self):
        names = [spec.name for spec in build_algorithms(ablation=True, smoothers=True)]
        self.assertEqual(len(names), len(set(names)))


class TestRecording(unittest.TestCase):
    def setUp(self):
        self.scores = {
            "success": True,
            "path_length": 12.3456,
            "smoothness": 0.123456,
            "curvature": 0.0654321,
            "max_curvature": 0.5,
            "clearance_min": 1.5,
            "clearance_mean": 2.5,
            "total_turning": 1.5708,
            "sharp_turns": 1,
            "radius_violated": True,
            "drive_time_curved": 2.0,
        }

    def test_a_static_row_fills_exactly_the_columns_the_file_has(self):
        row = static_row("dense_city", "Modified", 3, 0, comp_time=0.5, switches=2, collision_checks=120, astar_length=10.0, detection=float("nan"), scores=self.scores)

        # A row with a spare key or a missing one writes a file that reads back wrong
        self.assertEqual(set(row), set(STATIC_FIELDS))

    def test_the_length_ratio_is_the_length_over_the_optimum(self):
        row = static_row("dense_city", "Modified", 3, 0, 0.5, 2, 120, 10.0, float("nan"), self.scores)
        self.assertAlmostEqual(row["length_ratio"], 1.2346, places=4)
        self.assertEqual(row["astar_length"], 10.0)

    def test_an_escape_on_the_normal_city_is_a_false_alarm(self):
        # Nothing on a normal city map should trap APF, so any escape there is the detector firing when it should not
        alarm = static_row("normal_city", "Modified", 3, 0, 0.5, 1, 120, 10.0, float("nan"), self.scores)
        quiet = static_row("normal_city", "Modified", 3, 0, 0.5, 0, 120, 10.0, float("nan"), self.scores)
        elsewhere = static_row("blocked_road", "Modified", 3, 0, 0.5, 1, 120, 10.0, float("nan"), self.scores)

        self.assertEqual(alarm["false_alarm"], 1)
        self.assertEqual(quiet["false_alarm"], 0)
        self.assertTrue(np.isnan(elsewhere["false_alarm"]))

    def test_success_is_stored_as_a_number(self):
        row = static_row("dense_city", "Modified", 3, 0, 0.5, 2, 120, 10.0, float("nan"), self.scores)

        # The column is averaged into a success rate, so it has to be 1 and 0 rather than True and False
        self.assertEqual(row["success"], 1)

    def test_a_dynamic_row_fills_exactly_the_columns_the_file_has(self):
        result = {
            "success": False, "reached": False, "collided": True,
            "iters": 44, "replans": 2, "path": None, "switches": 0,
        }
        row = dynamic_row(1, 1.5, "RRT* replan", result, comp_time=0.25, collision_checks=120, path_length=30.0)

        self.assertEqual(set(row), set(DYNAMIC_FIELDS))
        self.assertEqual(row["collided"], 1)
        self.assertEqual(row["speed_factor"], 1.5)

    def test_the_file_reads_back_as_it_was_written(self):
        rows = [static_row("dense_city", "Modified", i, 0, 0.5, 1, 120, 10.0, float("nan"), self.scores) for i in range(3)]

        with tempfile.TemporaryDirectory() as folder:
            path = save_rows(rows, Path(folder) / "results.csv", STATIC_FIELDS)

            with open(path) as handle:
                written = list(csv.DictReader(handle))

            self.assertEqual(len(written), 3)
            self.assertEqual(list(written[0]), STATIC_FIELDS)


class TestAnalysis(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(
            [
                {"scenario": "s", "algorithm": "A", "instance": 0, "success": 1,
                 "path_length": 100.0, "smoothness": 0.2, "curvature": 0.05,
                 "max_curvature": 0.4, "clearance_min": 1.0, "clearance_mean": 2.0,
                 "comp_time": 0.1, "switches": 0},
                {"scenario": "s", "algorithm": "A", "instance": 1, "success": 0,
                 "path_length": 0.0, "smoothness": 0.0, "curvature": 0.0,
                 "max_curvature": 0.0, "clearance_min": 0.0, "clearance_mean": 0.0,
                 "comp_time": 0.3, "switches": 0},
            ]
        )

    def test_the_summary_table_reports_minimum_clearance(self):
        from planning.experiments.analyse import METRICS, summary_table

        # The summary reads every column in METRICS, so the ones this small frame does not carry are filled in
        frame = self.frame()
        for column in METRICS:
            if column not in frame:
                frame[column] = 0.0
        summary = summary_table(frame)

        # The one measurement that separates APF from RRT* has to be in the table the chapter is written from, and it has to be the successful run's value, not an average that includes the failed run's zero
        self.assertIn(("clearance_min", "median"), summary.columns)
        self.assertAlmostEqual(summary.loc[("s", "A"), ("clearance_min", "median")], 1.0)

    def test_the_summary_table_reports_medians_and_iqr_not_means(self):
        from planning.experiments.analyse import METRICS, RATES, summary_table

        rows = []
        for instance, length in enumerate((10.0, 12.0, 13.0, 14.0, 100.0)):
            rows.append({"scenario": "s", "algorithm": "A", "instance": instance, "success": 1, "path_length": length})
        frame = pd.DataFrame(rows)
        for column in METRICS:
            if column not in frame:
                frame[column] = 0.0
        summary = summary_table(frame)

        # One outlier of a hundred cells would drag a mean to 30; the median says 13 and the middle half spans 12 to 14
        self.assertAlmostEqual(summary.loc[("s", "A"), ("path_length", "median")], 13.0)
        self.assertAlmostEqual(summary.loc[("s", "A"), ("path_length", "iqr")], 2.0)

        # No skewed measurement keeps a mean or a standard deviation; only the rates are means
        for metric, statistic in summary.columns:
            if metric in RATES:
                self.assertEqual(statistic, "mean")
            else:
                self.assertIn(statistic, ("median", "iqr"))

    def test_the_tests_report_medians(self):
        result = paired_tests(self.frame(), "path_length", baseline="Modified")
        self.assertIn("baseline_median", result.columns)
        self.assertNotIn("baseline_mean", result.columns)

    def test_minimum_clearance_has_its_own_chart(self):
        from planning.experiments.plot import CHARTS, bar_chart

        self.assertIn("clearance_min", [chart[0] for chart in CHARTS])

        with tempfile.TemporaryDirectory() as folder:
            written = bar_chart(self.frame(), "clearance_min", "cells", "test", Path(folder) / "chart.png")
            self.assertTrue(Path(written).exists())

    def test_a_failed_run_is_left_out_of_the_route_averages(self):
        masked = mask_failures(self.frame())

        # Averaging a failed run's zero length against a real one would make giving up look like the shortest route
        self.assertEqual(masked["path_length"].mean(), 100.0)

    def test_a_failed_run_still_counts_against_the_success_rate(self):
        masked = mask_failures(self.frame())
        self.assertEqual(masked["success"].mean(), 0.5)

    def test_both_mission_times_are_derived_on_read(self):
        frame = pd.DataFrame([
            {"scenario": "normal_city", "algorithm": "APF", "instance": 0, "success": 1,
             "path_length": 83.0, "comp_time": 0.5, "drive_time_curved": 20.0},
        ])
        frame = add_mission_time(frame, speed=8.3)

        self.assertAlmostEqual(frame["mission_time"][0], 0.5 + 10.0, places=3)
        self.assertAlmostEqual(frame["mission_time_curved"][0], 0.5 + 20.0, places=3)

    def test_a_failed_run_keeps_the_time_it_spent(self):
        masked = mask_failures(self.frame())

        # The planner really did spend that time before giving up, so hiding it would flatter a slow algorithm that fails often
        self.assertAlmostEqual(masked["comp_time"].mean(), 0.2)

    def test_the_effect_size_is_one_when_one_side_always_wins(self):
        self.assertAlmostEqual(rank_biserial([5, 6, 7], [1, 2, 3]), 1.0)

    def test_the_effect_size_is_minus_one_when_it_always_loses(self):
        self.assertAlmostEqual(rank_biserial([1, 2, 3], [5, 6, 7]), -1.0)

    def test_ties_carry_no_effect(self):
        self.assertAlmostEqual(rank_biserial([1, 2, 3], [1, 2, 3]), 0.0)

    def test_the_tests_only_compare_instances_both_algorithms_solved(self):
        rows = []
        for instance in range(5):
            rows.append({"scenario": "s", "algorithm": "Modified", "instance": instance,
                         "success": 1, "path_length": 10.0 + instance})
            # The other algorithm fails on the last two maps
            rows.append({"scenario": "s", "algorithm": "APF", "instance": instance,
                         "success": 1 if instance < 3 else 0,
                         "path_length": 20.0 + instance if instance < 3 else 0.0})

        result = paired_tests(pd.DataFrame(rows), "path_length", baseline="Modified")

        # Comparing the length of a route that was never found against one that was is meaningless. The table says how many maps survived out of how many there were, so the reader can see the comparison is over the easier maps
        self.assertEqual(int(result["n_pairs"].iloc[0]), 3)
        self.assertEqual(int(result["n_maps"].iloc[0]), 5)

    def test_the_three_pre_specified_comparisons_are_labelled_and_the_rest_are_descriptive(self):
        rows = []
        for algorithm in ("Modified", "Hybrid", "APF", "RRT*", "Bezier", "Dubins"):
            for instance in range(4):
                rows.append({"scenario": "s", "algorithm": algorithm, "instance": instance,
                             "success": 1, "path_length": 10.0 + instance + len(algorithm)})

        result = paired_tests(pd.DataFrame(rows), "path_length", baseline="Modified").set_index("vs")

        for other in ("Hybrid", "APF", "RRT*"):
            self.assertEqual(result.loc[other, "kind"], "confirmatory")
        for other in ("Bezier", "Dubins"):
            self.assertEqual(result.loc[other, "kind"], "descriptive")

    def test_a_comparison_from_another_baseline_is_descriptive(self):
        rows = []
        for algorithm in ("Hybrid", "APF"):
            for instance in range(4):
                rows.append({"scenario": "s", "algorithm": algorithm, "instance": instance,
                             "success": 1, "path_length": 10.0 + instance + len(algorithm)})

        result = paired_tests(pd.DataFrame(rows), "path_length", baseline="Hybrid")
        self.assertEqual(result["kind"].iloc[0], "descriptive")


class TestRatesAndMcNemar(unittest.TestCase):
    def test_wilson_on_no_successes_still_has_an_upper_bound(self):
        low, high = wilson_interval(0, 10)
        self.assertAlmostEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.2775, places=3)

    def test_wilson_on_half(self):
        low, high = wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.2366, places=3)
        self.assertAlmostEqual(high, 0.7634, places=3)

    def test_wilson_on_nothing(self):
        low, high = wilson_interval(0, 0)
        self.assertTrue(np.isnan(low) and np.isnan(high))

    def test_mcnemar_uses_the_exact_form_under_twenty_five_discordant_pairs(self):
        # Five maps the baseline solved and the other did not, none the other way. Two-sided exact binomial on 0 of 5 is 2 times 0.5 to the fifth
        base = np.array([1, 1, 1, 1, 1, 1, 0])
        other = np.array([0, 0, 0, 0, 0, 1, 0])
        p_value, base_only, other_only, form = mcnemar(base, other)

        self.assertEqual((base_only, other_only), (5, 0))
        self.assertEqual(form, "mcnemar exact")
        self.assertAlmostEqual(p_value, 0.0625)

    def test_mcnemar_switches_to_chi_square_at_twenty_five(self):
        base = np.array([1] * 20 + [0] * 5)
        other = np.array([0] * 20 + [1] * 5)
        p_value, base_only, other_only, form = mcnemar(base, other)

        self.assertEqual((base_only, other_only), (20, 5))
        self.assertEqual(form, "mcnemar chi2")
        # (|20 - 5| - 1)^2 / 25 = 7.84 on one degree of freedom
        self.assertAlmostEqual(p_value, 0.0051, places=3)

    def test_no_discordant_pairs_is_no_evidence(self):
        p_value, _, _, _ = mcnemar(np.array([1, 1, 0]), np.array([1, 1, 0]))
        self.assertTrue(np.isnan(p_value))

    def frame(self):
        rows = []
        for instance in range(8):
            rows.append({"scenario": "s", "algorithm": "Modified", "instance": instance,
                         "success": 1, "path_length": 10.0})
            rows.append({"scenario": "s", "algorithm": "APF", "instance": instance,
                         "success": 1 if instance < 3 else 0, "path_length": 12.0})
        return pd.DataFrame(rows)

    def test_success_is_tested_with_mcnemar_and_never_wilcoxon(self):
        result = paired_tests(self.frame(), "success", baseline="Modified")

        self.assertEqual(list(result["test"]), ["mcnemar exact"])
        self.assertEqual(int(result["base_only"].iloc[0]), 5)
        self.assertEqual(int(result["other_only"].iloc[0]), 0)
        self.assertTrue(np.isnan(result["effect_size"].iloc[0]))

    def test_other_metrics_still_use_wilcoxon(self):
        result = paired_tests(self.frame(), "path_length", baseline="Modified")
        self.assertEqual(list(result["test"]), ["wilcoxon"])

    def test_every_rate_carries_its_interval_and_map_count(self):
        rates = success_rates(self.frame())
        apf = rates[rates["algorithm"] == "APF"].iloc[0]

        self.assertEqual(int(apf["n_maps"]), 8)
        self.assertAlmostEqual(apf["success"], 3 / 8)
        self.assertLess(apf["success_low"], apf["success"])
        self.assertGreater(apf["success_high"], apf["success"])


class TestAblationTable(unittest.TestCase):
    def frame(self):
        # Four cells on four maps, each a constant plus the map number so the effects are exact
        levels = {"Hybrid": 10.0, "Adaptive": 8.0, "Bezier": 6.0, "Modified": 2.0}
        rows = []
        for algorithm, level in levels.items():
            for instance in range(4):
                rows.append({"scenario": "s", "algorithm": algorithm, "instance": instance,
                             "success": 1, "max_curvature": level + instance})
        return pd.DataFrame(rows)

    def test_main_effects_and_interaction_come_from_the_four_cells(self):
        from planning.experiments.analyse import ablation_table

        table = ablation_table(self.frame(), metrics=("max_curvature",)).iloc[0]

        # Adaptive on: (8 + 2) / 2 against (10 + 6) / 2. Smoothing on: (6 + 2) / 2 against (10 + 8) / 2. Interaction: half the difference of the two differences
        self.assertAlmostEqual(table["effect_adaptive"], -3.0)
        self.assertAlmostEqual(table["effect_smoothing"], -5.0)
        self.assertAlmostEqual(table["interaction"], -1.0)
        self.assertEqual(int(table["n_pairs"]), 4)
        self.assertAlmostEqual(table["cell_adaptive_bezier"], 2.0 + 1.5)

    def test_success_is_read_as_a_rate_over_every_map(self):
        from planning.experiments.analyse import ablation_table

        frame = self.frame()
        frame.loc[(frame["algorithm"] == "Hybrid") & (frame["instance"] < 2), "success"] = 0
        table = ablation_table(frame, metrics=("success",)).iloc[0]

        self.assertAlmostEqual(table["cell_fixed_none"], 0.5)
        self.assertAlmostEqual(table["cell_adaptive_bezier"], 1.0)
        self.assertAlmostEqual(table["effect_adaptive"], 0.25)
        self.assertEqual(int(table["n_pairs"]), 4)

    def test_without_the_four_cells_there_is_no_table(self):
        from planning.experiments.analyse import ablation_table

        frame = self.frame()
        frame = frame[frame["algorithm"] != "Bezier"]
        self.assertEqual(len(ablation_table(frame, metrics=("max_curvature",))), 0)


class TestRepeats(unittest.TestCase):
    def frame(self):
        # Two maps, three repeats of a stochastic planner and one of APF. Map 0 succeeds every time with lengths 10, 12, 30; map 1 fails on two repeats of three
        rows = []
        for k, length in enumerate((10.0, 12.0, 30.0)):
            rows.append({"scenario": "s", "algorithm": "RRT*", "instance": 0, "repeat": k, "success": 1,
                         "path_length": length, "comp_time": 1.0, "switches": 0})
        for k, success in enumerate((0, 1, 0)):
            rows.append({"scenario": "s", "algorithm": "RRT*", "instance": 1, "repeat": k, "success": success,
                         "path_length": 20.0 if success else 0.0, "comp_time": 1.0, "switches": 0})
        rows.append({"scenario": "s", "algorithm": "APF", "instance": 0, "repeat": 0, "success": 1,
                     "path_length": 15.0, "comp_time": 0.1, "switches": 0})
        rows.append({"scenario": "s", "algorithm": "APF", "instance": 1, "repeat": 0, "success": 1,
                     "path_length": 25.0, "comp_time": 0.1, "switches": 0})
        return pd.DataFrame(rows)

    def test_the_paired_value_is_the_per_map_median_over_repeats(self):
        from planning.experiments.analyse import collapse_repeats

        collapsed = collapse_repeats(self.frame()).set_index(["algorithm", "instance"])

        # One row per map, the median of the three lengths, not their mean
        self.assertEqual(len(collapsed), 4)
        self.assertAlmostEqual(collapsed.loc[("RRT*", 0), "path_length"], 12.0)
        self.assertEqual(int(collapsed.loc[("RRT*", 0), "n_repeats"]), 3)
        self.assertEqual(int(collapsed.loc[("APF", 0), "n_repeats"]), 1)

    def test_success_on_a_map_is_the_majority_of_its_repeats(self):
        from planning.experiments.analyse import collapse_repeats

        collapsed = collapse_repeats(self.frame()).set_index(["algorithm", "instance"])
        self.assertEqual(int(collapsed.loc[("RRT*", 1), "success"]), 0)
        self.assertEqual(int(collapsed.loc[("RRT*", 0), "success"]), 1)

    def test_a_failed_repeat_does_not_drag_the_map_length_to_zero(self):
        from planning.experiments.analyse import collapse_repeats

        collapsed = collapse_repeats(self.frame()).set_index(["algorithm", "instance"])
        # Only the one successful repeat has a length, and that is the map's length
        self.assertAlmostEqual(collapsed.loc[("RRT*", 1), "path_length"], 20.0)

    def test_reliability_reports_the_within_map_spread(self):
        from planning.experiments.analyse import reliability_table

        table = reliability_table(self.frame()).set_index("algorithm")

        # The spread of 10, 12, 30 is the interquartile range 10; the one-repeat map has none. Map 1 disagreed with itself on success, map 0 did not
        self.assertAlmostEqual(table.loc["RRT*", "path_length_within_iqr"], 10.0)
        self.assertAlmostEqual(table.loc["RRT*", "success_disagreement"], 0.5)
        self.assertNotIn("APF", table.index)

    def test_a_file_without_repeats_collapses_to_itself(self):
        from planning.experiments.analyse import collapse_repeats

        frame = self.frame()
        frame = frame[frame["repeat"] == 0].drop(columns="repeat")
        collapsed = collapse_repeats(frame)
        self.assertEqual(len(collapsed), 4)

    def test_the_runner_runs_every_repeat_of_a_stochastic_planner_and_one_of_apf(self):
        config = tiny_config(n_repeats=3, scenarios=("normal_city",), n_instances=1)
        frame = pd.DataFrame(StaticStudy(config, build_algorithms()).run())

        counts = frame.groupby("algorithm")["repeat"].nunique()
        self.assertEqual(int(counts["APF"]), 1)
        self.assertEqual(int(counts["RRT*"]), 3)
        self.assertEqual(sorted(frame[frame["algorithm"] == "RRT*"]["repeat"]), [0, 1, 2])


class TestDynamicStudySpeeds(unittest.TestCase):
    def test_every_driver_meets_every_speed_on_every_walk(self):
        config = tiny_config(
            dynamic_speed_factors=(0.5, 1.5), dynamic_max_steps=200, rrt_max_iter=600
        )
        study = DynamicStudy(config, n_instances=2)
        rows, examples = study.run()
        frame = pd.DataFrame(rows)

        # Two walks, two speeds, five drivers
        self.assertEqual(len(frame), 2 * 2 * 5)
        self.assertEqual(set(frame["speed_factor"]), {0.5, 1.5})
        counts = frame.groupby(["instance", "speed_factor"])["driver"].nunique()
        self.assertTrue((counts == 5).all())

        # One example per driver for the animations, whichever speed it came from
        self.assertEqual(len(examples), 5)

    def test_the_study_size_comes_from_the_configuration(self):
        config = tiny_config(dynamic_instances=7)
        self.assertEqual(DynamicStudy(config).n_instances, 7)


class TestStaticStudyEndToEnd(unittest.TestCase):
    def test_the_loop_produces_one_row_per_run(self):
        config = tiny_config()
        algorithms = build_algorithms()

        rows = StaticStudy(config, algorithms).run()

        expected = len(config.scenarios) * config.n_instances * len(algorithms)
        self.assertEqual(len(rows), expected)

    def test_every_algorithm_meets_every_map(self):
        config = tiny_config()
        algorithms = build_algorithms()

        frame = pd.DataFrame(StaticStudy(config, algorithms).run())

        # The paired design falls apart the moment one algorithm is missing a map
        counts = frame.groupby(["scenario", "instance"])["algorithm"].nunique()
        self.assertTrue((counts == len(algorithms)).all())

    def test_workers_change_nothing_but_the_timing(self):
        config = tiny_config()

        sequential = pd.DataFrame(StaticStudy(config, build_algorithms()).run())
        parallel = pd.DataFrame(StaticStudy(config, build_algorithms(), workers=2).run())

        # Every run is seeded from the master seed rather than drawn from a shared stream, so spreading the maps across processes has to give back the same results
        columns = [c for c in sequential.columns if c != "comp_time"]
        pd.testing.assert_frame_equal(
            sequential[columns].reset_index(drop=True),
            parallel[columns].reset_index(drop=True),
        )

    def test_the_check_count_is_recorded_and_repeats_itself(self):
        config = tiny_config()
        study = StaticStudy(config, build_algorithms())

        # Three runs of one map, as the budget column has to be a property of the seed and not of the wall clock
        runs = [pd.DataFrame(study.run_instance("blocked_road", 0)) for _ in range(3)]

        self.assertTrue((runs[0]["collision_checks"] > 0).all())
        for later in runs[1:]:
            pd.testing.assert_series_equal(runs[0]["collision_checks"], later["collision_checks"])

    def test_the_scorer_does_not_bill_the_planner(self):
        # The scorer validates the route with the same helpers the planners use. Its checks must not land in the planner's column, so a run's count has to be what the planner did on its own
        from planning.utils.collision import checks_done, reset_checks

        config = tiny_config()
        study = StaticStudy(config, build_algorithms())
        spec = build_algorithms()[0]
        grid, start, goal = make_instance("normal_city", config.master_seed, config)

        reset_checks()
        result, elapsed, billed = study.run_one(spec, grid, start, goal, seed=1)
        self.assertEqual(billed, checks_done())

    def test_no_successful_run_beats_the_optimum(self):
        # The ratio is only a yardstick if nothing gets under it. A success below 1.0 would mean the A* rule is looser than the scorer's
        config = tiny_config()
        frame = pd.DataFrame(StaticStudy(config, build_algorithms()).run())
        successes = frame[frame["success"] == 1]

        self.assertGreater(len(successes), 0)
        self.assertGreaterEqual(successes["length_ratio"].min(), 1.0)

    def test_the_blocked_road_rows_carry_a_time_to_detection(self):
        config = tiny_config()
        frame = pd.DataFrame(StaticStudy(config, build_algorithms()).run())

        blocked = frame[(frame["scenario"] == "blocked_road") & (frame["algorithm"] == "Hybrid")]
        normal = frame[frame["scenario"] == "normal_city"]

        # Both columns exist on every row; the detection time is only ever a number on the blocked road, and the false alarm only ever on the normal city
        self.assertIn("time_to_detection", frame.columns)
        self.assertTrue(normal["time_to_detection"].isna().all())
        self.assertTrue(frame[frame["scenario"] == "blocked_road"]["false_alarm"].isna().all())
        self.assertFalse(normal["false_alarm"].isna().any())

    def test_smoothing_does_not_change_the_time_to_detection(self):
        # The APF phase before the first escape is the same with and without smoothing, so the detection time has to be too. It was not, when the count was taken on the smoothed path
        config = tiny_config()
        frame = pd.DataFrame(StaticStudy(config, build_algorithms(ablation=True)).run())
        blocked = frame[frame["scenario"] == "blocked_road"].set_index(["algorithm", "instance"])

        for instance in range(config.n_instances):
            plain = blocked.loc[("Hybrid", instance), "time_to_detection"]
            smoothed = blocked.loc[("Bezier", instance), "time_to_detection"]
            with self.subTest(instance=instance):
                if np.isnan(plain):
                    self.assertTrue(np.isnan(smoothed))
                else:
                    self.assertEqual(plain, smoothed)

    def test_the_whole_study_repeats_itself(self):
        config = tiny_config()

        first = pd.DataFrame(StaticStudy(config, build_algorithms()).run())
        second = pd.DataFrame(StaticStudy(config, build_algorithms()).run())

        # Everything except the wall clock timing has to come back identical
        columns = [c for c in first.columns if c != "comp_time"]
        pd.testing.assert_frame_equal(first[columns], second[columns])


if __name__ == "__main__":
    unittest.main()
