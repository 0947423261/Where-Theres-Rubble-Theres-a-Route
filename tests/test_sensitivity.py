"""
tests/test_sensitivity.py

The power estimate and the pilot it is run on. The pilot decides how many maps the real run gets, so the estimate has to detect a difference that is there, stay quiet when there is none, and be run on a seed the real experiment never uses.
"""

import unittest

import numpy as np
import pandas as pd

from planning.experiments.sensitivity import (
    PILOT_INSTANCES,
    PILOT_SEED,
    POWER_COMPARISONS,
    TUNING_GRID,
    TUNING_INSTANCES,
    TUNING_SEED,
    bootstrap_power,
    choose_setting,
    pilot_config,
    PLANNER_SWEEPS,
    run_planner_sweeps,
    smallest_adequate_size,
    summarise_sweep,
    sweep_goal_offset,
    sweep_parameter,
    to_long,
    tune_planner,
    tuning_config,
)
from planning.experiments.analyse import CONFIRMATORY
from planning.utils.config import Config


def paired_frame(base_values, other_values, base_success=None, other_success=None):
    """Two algorithms on one scenario, one row per map each."""
    rows = []
    for i, (b, o) in enumerate(zip(base_values, other_values)):
        rows.append({"scenario": "s", "algorithm": "Modified", "instance": i, "metric": b,
                     "success": 1 if base_success is None else base_success[i]})
        rows.append({"scenario": "s", "algorithm": "APF", "instance": i, "metric": o,
                     "success": 1 if other_success is None else other_success[i]})
    return pd.DataFrame(rows)


class TestBootstrapPower(unittest.TestCase):
    def test_a_difference_on_every_map_is_found_at_a_small_size(self):
        frame = paired_frame([10.0 + i for i in range(12)], [14.0 + i for i in range(12)])
        power = bootstrap_power(frame, "s", "metric", "Modified", "APF", sizes=(10,), trials=100)

        self.assertEqual(power["test"].iloc[0], "wilcoxon")
        self.assertGreaterEqual(power["power"].iloc[0], 0.95)

    def test_no_difference_is_not_found(self):
        frame = paired_frame([10.0 + i for i in range(12)], [10.0 + i for i in range(12)])
        power = bootstrap_power(frame, "s", "metric", "Modified", "APF", sizes=(10, 30), trials=100)

        self.assertTrue((power["power"] == 0.0).all())

    def test_success_is_resampled_as_paired_outcomes_under_mcnemar(self):
        # The baseline solves every map and the other solves none: every map is discordant one way. Ten resampled maps give McNemar exact p of about 0.002
        frame = paired_frame([1] * 12, [0] * 12, base_success=[1] * 12, other_success=[0] * 12)
        power = bootstrap_power(frame, "s", "success", "Modified", "APF", sizes=(10,), trials=100)

        self.assertEqual(power["test"].iloc[0], "mcnemar")
        self.assertGreaterEqual(power["power"].iloc[0], 0.95)

    def test_success_with_few_discordant_maps_needs_more_maps(self):
        # Two maps in twelve where only the baseline succeeds. At ten maps most resamples hold one or two discordant pairs, which McNemar cannot call
        base = [1] * 12
        other = [1] * 10 + [0, 0]
        frame = paired_frame(base, other, base_success=base, other_success=other)
        power = bootstrap_power(frame, "s", "success", "Modified", "APF", sizes=(10, 100), trials=200)

        self.assertLess(power.set_index("n_instances").loc[10, "power"], 0.3)
        self.assertGreater(power.set_index("n_instances").loc[100, "power"], power.set_index("n_instances").loc[10, "power"])

    def test_quality_is_compared_on_the_maps_both_solved(self):
        frame = paired_frame([10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 0.0, 0.0], other_success=[1, 1, 0, 0])
        power = bootstrap_power(frame, "s", "metric", "Modified", "APF", sizes=(10,), trials=10)

        self.assertEqual(int(power["n_pairs_observed"].iloc[0]), 2)


class TestSmallestAdequateSize(unittest.TestCase):
    def test_the_first_size_at_or_over_the_target(self):
        table = pd.DataFrame({"n_instances": [10, 20, 30], "power": [0.4, 0.85, 0.99]})
        self.assertEqual(smallest_adequate_size(table, 0.8), 20)

    def test_none_when_no_size_reaches_it(self):
        table = pd.DataFrame({"n_instances": [10, 20], "power": [0.1, 0.2]})
        self.assertIsNone(smallest_adequate_size(table, 0.8))


class TestPilot(unittest.TestCase):
    def test_the_pilot_runs_on_a_seed_the_experiment_never_uses(self):
        base = Config(out_dir="/tmp/test_outputs")
        pilot = pilot_config(base)

        self.assertEqual(pilot.master_seed, PILOT_SEED)
        self.assertNotEqual(pilot.master_seed, base.master_seed)
        self.assertEqual(pilot.n_instances, PILOT_INSTANCES)
        self.assertEqual(pilot.n_repeats, 1)

    def test_the_power_comparisons_are_the_confirmatory_pairs_on_the_tested_metrics(self):
        from planning.experiments.analyse import TESTED_METRICS

        # The run is sized for the tests it reports, so every power comparison is a confirmatory pair on a metric the significance table tests
        for _, metric, baseline, other in POWER_COMPARISONS:
            self.assertIn((baseline, other), CONFIRMATORY)
            self.assertIn(metric, TESTED_METRICS)


class TestTuning(unittest.TestCase):
    def test_tuning_runs_on_its_own_seed(self):
        base = Config(out_dir="/tmp/test_outputs")
        tuning = tuning_config(base)

        self.assertEqual(tuning.master_seed, TUNING_SEED)
        self.assertNotEqual(tuning.master_seed, base.master_seed)
        self.assertNotEqual(tuning.master_seed, PILOT_SEED)
        self.assertEqual(tuning.n_instances, TUNING_INSTANCES)
        self.assertEqual(tuning.n_repeats, 1)

    def test_every_planner_gets_the_same_budget_and_the_default_is_a_candidate(self):
        base = Config(out_dir="/tmp/test_outputs")
        budgets = set()

        for planner, grid in TUNING_GRID.items():
            # The budget is the number of distinct settings evaluated: the default once, plus every other value of every parameter
            budgets.add(sum(len(values) - 1 for values in grid.values()) + 1)
            for parameter, values in grid.items():
                self.assertIn(getattr(base, parameter), values, (planner, parameter))

        self.assertEqual(len(budgets), 1)

    def test_the_choice_is_success_first_then_the_drive(self):
        rows = []
        # Three settings on four maps. 1.0 solves three and drives slowly, 2.0 solves three and drives fast, 3.0 solves one and drives fastest. Two maps short is out of the running whatever its drive
        for value, successes, drive in ((1.0, 3, 30.0), (2.0, 3, 20.0), (3.0, 1, 10.0)):
            for instance in range(4):
                rows.append({"parameter": "p", "value": value, "instance": instance, "scenario": "s",
                             "success": 1 if instance < successes else 0,
                             "drive_time_curved": drive if instance < successes else 0.0})
        frame = pd.DataFrame(rows)

        chosen, table = choose_setting(frame, "p", default=1.0)

        self.assertEqual(chosen, 2.0)
        self.assertEqual(len(table), 3)

    def test_a_setting_one_map_short_on_success_can_still_win_on_the_drive(self):
        rows = []
        # Twenty maps. 1.0 solves all twenty and drives slowly; 2.0 solves nineteen and drives much faster. One map is inside the tolerance, so the faster one wins
        for value, successes, drive in ((1.0, 20, 30.0), (2.0, 19, 20.0)):
            for instance in range(20):
                rows.append({"parameter": "p", "value": value, "instance": instance, "scenario": "s",
                             "success": 1 if instance < successes else 0,
                             "drive_time_curved": drive if instance < successes else 0.0})

        chosen, _ = choose_setting(pd.DataFrame(rows), "p", default=1.0)
        self.assertEqual(chosen, 2.0)

    def test_one_map_short_on_sixty_is_still_a_contender(self):
        # Nineteen of sixty against eighteen of sixty. Compared as rounded rates the gap came out a hair over one map and the slower candidate won; compared in maps it is one map and the faster drive wins
        rows = []
        for value, successes, drive in ((1.0, 19, 100.0), (2.0, 18, 68.0)):
            for instance in range(60):
                rows.append({"parameter": "p", "value": value, "instance": instance, "scenario": "s",
                             "success": 1 if instance < successes else 0,
                             "drive_time_curved": drive if instance < successes else 0.0})

        chosen, _ = choose_setting(pd.DataFrame(rows), "p", default=2.0)
        self.assertEqual(chosen, 2.0)

    def test_a_tie_keeps_the_default(self):
        rows = []
        for value in (1.0, 2.0):
            for instance in range(4):
                rows.append({"parameter": "p", "value": value, "instance": instance, "scenario": "s",
                             "success": 1, "drive_time_curved": 10.0})

        chosen, _ = choose_setting(pd.DataFrame(rows), "p", default=1.0)
        self.assertEqual(chosen, 1.0)

    def test_tune_planner_evaluates_the_grid_and_picks_from_it(self):
        base = Config(
            grid_size=30, start=(2, 3), goal=(27, 26), n_instances=1, n_repeats=1,
            scenarios=("normal_city",), rrt_max_iter=600, out_dir="/tmp/test_outputs",
        )
        grid = {"rrt_step": (4.0, 6.0)}

        chosen, rows = tune_planner("RRT*", base, grid, workers=1)

        self.assertIn(chosen["rrt_step"], (4.0, 6.0))
        self.assertEqual(set(rows["value"]), {4.0, 6.0})
        self.assertEqual(set(rows["planner"]), {"RRT*"})


class TestGoalOffsetSweep(unittest.TestCase):
    def small(self):
        return Config(grid_size=30, start=(2, 3), goal=(27, 26), n_repeats=1, out_dir="/tmp/test_outputs")

    def test_the_goal_moves_off_the_diagonal_by_the_offset(self):
        from planning.experiments.runner import build_algorithms

        apf = [spec for spec in build_algorithms() if spec.name == "APF"]
        rows = sweep_goal_offset(self.small(), offsets=(0, 2), n_instances=1, algorithms=apf, scenarios=("normal_city",))

        # The value column is the offset, and each offset was run on the same map
        self.assertEqual(set(rows["value"]), {0, 2})
        self.assertEqual(set(rows["parameter"]), {"goal"})
        self.assertEqual(set(rows["algorithm"]), {"APF"})
        self.assertIn("drive_time_curved", rows.columns)

    def test_value_labels_must_match_the_values(self):
        with self.assertRaises(ValueError):
            sweep_parameter(self.small(), "goal", [(27, 27), (27, 25)], value_labels=[0], n_instances=1)

    def test_the_offset_chart_is_written(self):
        import tempfile
        from pathlib import Path
        from planning.experiments.plot import plot_goal_offset

        rows = []
        for offset in (0, 1, 3):
            for algorithm in ("APF", "RRT*"):
                for instance in range(3):
                    rows.append({"parameter": "goal", "value": offset, "scenario": "s", "algorithm": algorithm,
                                 "instance": instance, "success": 1 if instance < 2 else 0,
                                 "drive_time_curved": 20.0 + offset if instance < 2 else 0.0})

        with tempfile.TemporaryDirectory() as folder:
            written = plot_goal_offset(pd.DataFrame(rows), Path(folder) / "chart.png")
            self.assertTrue(Path(written).exists())


def dummy_sweep():
    """Two values of one parameter, two planners, three maps, in the wide form sweep_parameter returns."""
    rows = []
    for value in (1.0, 2.0):
        for algorithm, base in (("APF", 10.0), ("RRT*", 20.0)):
            for instance in range(3):
                success = 0 if (algorithm == "APF" and instance == 2) else 1
                rows.append({"scenario": "s", "algorithm": algorithm, "instance": instance,
                             "parameter": "p", "value": value,
                             "success": success,
                             "drive_time_curved": (base + value + instance) if success else 0.0,
                             "comp_time": 0.1 * value})
    return pd.DataFrame(rows)


class TestLongFormat(unittest.TestCase):
    def test_one_row_per_planner_scenario_map_parameter_value_and_metric(self):
        wide = dummy_sweep()
        long = to_long(wide)

        self.assertEqual(list(long.columns), ["planner", "scenario", "instance", "parameter", "value", "metric", "result"])
        # 12 wide rows, three metrics each
        self.assertEqual(len(long), 12 * 3)
        self.assertEqual(set(long["metric"]), {"success", "drive_time_curved", "comp_time"})

    def test_a_failed_run_has_no_route_result_but_keeps_its_time(self):
        long = to_long(dummy_sweep()).set_index(["planner", "instance", "value", "metric"])

        self.assertTrue(np.isnan(long.loc[("APF", 2, 1.0, "drive_time_curved"), "result"]))
        self.assertEqual(long.loc[("APF", 2, 1.0, "success"), "result"], 0)
        self.assertAlmostEqual(long.loc[("APF", 2, 1.0, "comp_time"), "result"], 0.1)

    def test_the_seed_set_is_shared_across_values(self):
        # Every value sees the same maps, which is what makes a difference between two values the parameter's doing
        long = to_long(dummy_sweep())
        maps_at = {value: set(block["instance"]) for value, block in long.groupby("value")}
        self.assertEqual(maps_at[1.0], maps_at[2.0])


class TestSummariseSweep(unittest.TestCase):
    def test_medians_and_quartiles_per_value_and_planner_with_failures_counted(self):
        table = summarise_sweep(to_long(dummy_sweep())).set_index(["planner", "value", "metric"])

        # RRT* at value 1 drove 21, 22, 23 over three maps
        row = table.loc[("RRT*", 1.0, "drive_time_curved")]
        self.assertAlmostEqual(row["median"], 22.0)
        self.assertAlmostEqual(row["q25"], 21.5)
        self.assertAlmostEqual(row["q75"], 22.5)
        self.assertEqual(int(row["n_maps"]), 3)
        self.assertEqual(int(row["n_excluded"]), 0)

        # APF failed one map, so its drive is over two and one was excluded
        row = table.loc[("APF", 1.0, "drive_time_curved")]
        self.assertEqual(int(row["n_excluded"]), 1)
        self.assertAlmostEqual(row["median"], 11.5)


class TestSweepPlot(unittest.TestCase):
    def test_one_chart_per_metric_with_a_band(self):
        import tempfile
        from pathlib import Path
        from planning.experiments.plot import plot_sweep

        long = to_long(dummy_sweep())
        with tempfile.TemporaryDirectory() as folder:
            written = plot_sweep(long, "p", ("success", "drive_time_curved"), Path(folder) / "sweep.png")
            self.assertTrue(Path(written).exists())


class TestDummySweepEndToEnd(unittest.TestCase):
    def test_a_two_value_sweep_writes_a_correct_long_file(self):
        import tempfile
        from pathlib import Path
        from planning.experiments.runner import build_algorithms
        from planning.experiments.sensitivity import run_sweep

        config = Config(grid_size=30, start=(2, 3), goal=(27, 26), n_repeats=1, rrt_max_iter=600, out_dir=tempfile.mkdtemp())
        apf = [spec for spec in build_algorithms() if spec.name == "APF"]

        written = run_sweep(config, "apf_rho0", (2.0, 3.0), n_instances=2, algorithms=apf, scenarios=("normal_city",), workers=1)
        long = pd.read_csv(written)

        self.assertEqual(Path(written).name, "sweep_apf_rho0.csv")
        self.assertEqual(list(long.columns), ["planner", "scenario", "instance", "parameter", "value", "metric", "result"])
        self.assertEqual(set(long["value"]), {2.0, 3.0})
        self.assertEqual(set(long["instance"]), {0, 1})
        # The same map at both values: the rows pair up
        counts = long.groupby(["value", "metric"]).size()
        self.assertTrue((counts == 2).all())
        self.assertTrue((Path(written).parent / "chart_sweep_apf_rho0.png").exists())


class TestPlannerSweeps(unittest.TestCase):
    def test_every_swept_parameter_is_a_config_field_holding_its_default(self):
        base = Config(out_dir="/tmp/test_outputs")
        for planner, grid in PLANNER_SWEEPS.items():
            for parameter, values in grid.items():
                self.assertIn(getattr(base, parameter), values, (planner, parameter))
                self.assertGreaterEqual(len(values), 3, (planner, parameter))

    def test_a_planner_sweep_writes_one_long_file_per_parameter(self):
        import tempfile
        from pathlib import Path

        config = Config(grid_size=30, start=(2, 3), goal=(27, 26), n_repeats=1, out_dir=tempfile.mkdtemp())
        written = run_planner_sweeps("APF", config, grid={"apf_rho0": (2.0, 3.0)}, n_instances=1, scenarios=("normal_city",), workers=1)

        self.assertEqual([Path(w).name for w in written], ["sweep_apf_rho0.csv"])
        long = pd.read_csv(written[0])
        self.assertEqual(set(long["planner"]), {"APF"})
        self.assertEqual(set(long["value"]), {2.0, 3.0})
