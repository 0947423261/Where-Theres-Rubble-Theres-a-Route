"""
experiments/sensitivity.py

Two questions this module answers, both of which a reader is entitled to ask about any tuned algorithm.

The first is whether the results survive the parameters being chosen differently. Every planner here has numbers that were tuned by hand, and a comparison that only holds at one setting is a comparison of settings rather than of algorithms. One parameter at a time is varied while the rest stay at their defaults, and each setting is rerun over the same maps.

The second is how many maps the experiment needs. Rather than assert a number, the number is estimated from a pilot: twenty maps per scenario on a throwaway master seed the real run never uses, and the observed paired differences resampled to see how often a run of a given size would detect them. Doing that on the final data would be circular, since the run would then be sized to find what it found.

The third is where the parameter values came from. Every planner is tuned on held-out maps from a seed the experiment never uses, with the same number of candidate settings for each, and the chosen values are locked in config.py with a comment naming the sweep. Tuning on the maps that produce the results would let the planners be fitted to them.

None of these is a metric of the planners. They are checks on the experiment, and they are kept apart from runner.py for that reason.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ..maps.scenarios import make_instance
from ..utils.config import Config
from ..utils.metrics import score_run
from .analyse import collapse_repeats, load, mcnemar
from .record import STATIC_FIELDS, save_rows
from .runner import StaticStudy, build_algorithms, derived_seed

# The pilot: a master seed the real experiment never uses, and how many maps per scenario it runs. The seed is a throwaway, chosen once and not looked at again
PILOT_SEED = 7
PILOT_INSTANCES = 20

# The comparisons the run is sized for, (scenario, metric, baseline, other). They are the three confirmatory pairs on the measurements the claims rest on: whether the modified hybrid gets through where APF does not, and whether it is quicker and turns more gently than the standard hybrid and than RRT*
POWER_COMPARISONS = (
    ("blocked_road", "success", "Modified", "APF"),
    ("normal_city", "success", "Modified", "APF"),
    ("normal_city", "mission_time_curved", "Modified", "Hybrid"),
    ("dense_city", "mission_time_curved", "Modified", "Hybrid"),
    ("normal_city", "max_curvature", "Modified", "Hybrid"),
    ("normal_city", "mission_time_curved", "Modified", "RRT*"),
    ("normal_city", "max_curvature", "Modified", "RRT*"),
)

# The run sizes the power is estimated at. Fifty is the floor the design allows and a hundred the target
POWER_SIZES = (10, 20, 30, 50, 75, 100)

# Tuning: a third throwaway seed, and how many held-out maps per scenario the candidates are judged on
TUNING_SEED = 11
TUNING_INSTANCES = 20

# The candidate settings each planner is tuned over, one parameter at a time around the value in config.py. Every planner has three parameters with three values each, so the budget is the same for all of them: the default once plus six other settings. The hybrids share the APF and RRT* parameters, which are tuned first on the standalone planners; the standard hybrid then tunes the escape parameters both hybrids use, and the modified hybrid tunes only the parameters of its own two extensions
TUNING_GRID = {
    "APF": {
        "apf_k_rep": (60.0, 120.0, 240.0),
        "apf_rho0": (2.0, 3.0, 4.5),
        "apf_step": (0.5, 0.7, 1.0),
    },
    "RRT*": {
        "rrt_step": (3.0, 4.0, 6.0),
        "rrt_radius": (6.0, 8.0, 12.0),
        "rrt_goal_bias": (0.05, 0.1, 0.2),
    },
    "Hybrid": {
        "stuck_window_fixed": (8, 12, 18),
        "stuck_delta": (1.5, 2.0, 3.0),
        "subgoal_dist": (12.0, 18.0, 27.0),
    },
    "Modified": {
        "stuck_window_min": (3, 4, 6),
        "density_radius": (4, 6, 9),
        "smooth_offset": (3, 4, 6),
    },
}

# The order the planners are tuned in, so a planner that reuses another's parameters sees them already tuned
TUNING_ORDER = ("APF", "RRT*", "Hybrid", "Modified")

# The parameters worth varying, each with the settings to try.
#
# Every one of these was tuned by hand, and each was picked because a reader could reasonably ask whether the choice is what produced the result:
#
#   apf_step        how far the vehicle moves per APF step. Suspected of driving the sharp turning, since a step that overshoots the equilibrium makes the vehicle zigzag across it
#   apf_rho0        how far the repulsion from an obstacle reaches. Sets how wide a channel APF needs, so it decides which streets are driveable at all
#   apf_k_rep       how hard obstacles push. Trades clearance against getting through a gap
#   rrt_step        the branch length. Sets how well the tree fits down a narrow street
#   stuck_delta     the displacement below which the hybrid calls itself stuck. Decides how eagerly it escapes
#   subgoal_dist    how far along the route an escape aims. Trades escape cost against progress
SWEEPS = {
    "apf_step": (0.4, 0.55, 0.7, 0.9),
    "apf_rho0": (2.0, 3.0, 4.0, 5.0),
    "apf_k_rep": (60.0, 120.0, 200.0),
    "rrt_step": (2.0, 3.0, 4.0, 6.0),
    "stuck_delta": (1.0, 2.0, 3.0),
    "subgoal_dist": (10.0, 18.0, 26.0),
}

# The columns a sweep reports, which are the ones a parameter could plausibly move
REPORTED = ["success", "path_length", "length_ratio", "curvature", "max_curvature", "drive_time_curved", "comp_time"]

# The one-at-a-time grids for the per-planner sweeps, coarse and wide around the locked values, so the plateaus and the cliffs show. Every grid holds the default. These are the sweeps that show how each planner depends on its settings; they are not a retune, which was JR19 on held-out maps
PLANNER_SWEEPS = {
    "APF": {
        "apf_k_att": (0.5, 1.0, 2.0, 4.0),
        "apf_k_rep": (30.0, 60.0, 120.0, 240.0, 480.0),
        "apf_rho0": (1.5, 2.0, 3.0, 4.5, 6.0),
        "apf_step": (0.35, 0.5, 0.7, 1.0, 1.4),
        "apf_max_iter": (1000, 2000, 4000, 8000),
    },
    "RRT*": {
        "rrt_step": (2.0, 3.0, 4.0, 6.0, 9.0),
        "rrt_goal_bias": (0.02, 0.05, 0.1, 0.2, 0.4),
        "rrt_radius": (6.0, 8.0, 12.0, 18.0),
        "rrt_max_iter": (1500, 3000, 6000, 12000),
        "rrt_goal_thresh": (2.0, 4.0, 8.0),
    },
    "Modified": {
        "stuck_window_fixed": (4, 8, 12, 18),
        "stuck_delta": (1.0, 2.0, 3.0, 4.5),
        "subgoal_dist": (9.0, 12.0, 18.0, 27.0, 40.0),
        "max_escapes": (3, 6, 12, 24),
        "stuck_window_min": (2, 4, 6, 10),
        "stuck_window_max": (12, 20, 30),
        "density_radius": (3, 6, 9, 12),
        "turn_threshold": (0.2, 0.35, 0.5, 0.8),
        "smooth_offset": (2, 4, 6, 9),
        "smooth_samples": (12, 24, 48),
    },
}

# The goal offsets swept for the goal-placement check, in cells off the corner-to-corner diagonal. Zero is the exact diagonal
GOAL_OFFSETS = (0, 1, 2, 3, 5)


def _paired_config(base, **overrides):
    """
    Build a configuration that differs from the base in the named settings only.

    The maps have to stay fixed while a parameter moves, otherwise a change in the results could be the parameter or could be the different maps. Everything the generator reads is copied across unchanged, so the same seed rebuilds the same city at every setting.
    """
    settings = {
        field: getattr(base, field)
        for field in base.__dataclass_fields__
        if field != "out_dir"
    }
    settings.update(overrides)

    return Config(out_dir=base.out_dir, **settings)


def sweep_parameter(base_config, parameter, values, algorithms=None, n_instances=6, scenarios=None, workers=1, value_labels=None):
    """
    Rerun the experiment at each value of one parameter and return a row per setting, scenario and algorithm.

    Only the named parameter moves. The maps, the seeds and every other setting stay where they were, so the difference between two rows is the parameter and nothing else. Each setting runs through the same harness as the main experiment, workers included, with one repeat per planner. value_labels, when given, is what the value column says for each value, for a parameter whose values do not read well in a table.
    """
    # The core planners unless the caller wants a different set. The ablation variants are left out because a sweep multiplies the runtime by the number of settings
    algorithms = algorithms if algorithms is not None else build_algorithms()

    # Dense city costs several minutes per setting because every failed RRT* escape spends its whole sample budget, and a sweep pays that for every value of the parameter. The two cheaper scenarios already show whether a result moves, so they are the default and dense city is opt in
    scenarios = scenarios if scenarios is not None else ("normal_city", "blocked_road")

    if value_labels is None:
        value_labels = list(values)
    if len(value_labels) != len(values):
        raise ValueError("value_labels must have one label per value")

    frames = []

    for value, label in zip(values, value_labels):
        config = _paired_config(
            base_config, n_instances=n_instances, n_repeats=1, scenarios=tuple(scenarios), **{parameter: value}
        )

        rows = pd.DataFrame(StaticStudy(config, algorithms, workers=workers).run())
        columns = ["scenario", "algorithm", "instance"] + [c for c in REPORTED if c in rows.columns]
        frames.append(rows[columns].assign(parameter=parameter, value=label))

        print(f"  {parameter} = {label} done", flush=True)

    return pd.concat(frames, ignore_index=True)


def run_planner_sweeps(name, config, grid=None, n_instances=30, scenarios=None, workers=1):
    """
    Sweep every parameter in one planner's grid, one long file and one chart per parameter, and return the paths of the files written.

    Thirty maps per value on every scenario by default, which is what the sensitivity skill asks for. Only the named planner runs, so the sweep answers how that planner depends on its own settings.
    """
    grid = grid if grid is not None else PLANNER_SWEEPS[name]
    scenarios = scenarios if scenarios is not None else tuple(config.scenarios)
    spec = [s for s in build_algorithms(ablation=True, smoothers=True) if s.name == name]

    written = []
    for parameter, values in grid.items():
        print(f"=== {name}: sweeping {parameter} over {values} ===", flush=True)
        written.append(run_sweep(
            config, parameter, values, n_instances=n_instances, algorithms=spec, scenarios=scenarios, workers=workers,
        ))

    return written


def sweep_goal_offset(base_config, offsets=GOAL_OFFSETS, n_instances=6, algorithms=None, scenarios=None, workers=1):
    """
    Sweep where the goal sits: on the corner-to-corner diagonal, then pulled off it by each offset in cells.

    The goal in config.py sits one cell off the diagonal on purpose, so that the pull towards it does not cancel exactly against the push off the first building corner it meets. This sweep is the check that the choice matters and that the results do not hinge on the exact cell chosen. The value column holds the offset.
    """
    n = base_config.grid_size
    goals = [(n - 3, n - 3 - offset) for offset in offsets]

    return sweep_parameter(
        base_config, "goal", goals, algorithms=algorithms, n_instances=n_instances,
        scenarios=scenarios, workers=workers, value_labels=list(offsets),
    )


# The measurements that only mean something for a run that arrived. A failed run has no route to measure, so its result is blank in the long file
ROUTE_METRICS = ("path_length", "length_ratio", "curvature", "max_curvature", "drive_time_curved")

# The metrics a sweep is judged on, in the order they are drawn
SWEEP_METRICS = ("success", "length_ratio", "drive_time_curved", "max_curvature")


def to_long(wide):
    """
    Turn a sweep's rows into the long format everything downstream reads: one row per (planner, scenario, instance, parameter, value, metric, result).

    Every measurement becomes its own row, so a new metric is a new row and not a new column, and a file from any sweep can be concatenated with any other. A route measurement on a failed run is blank; success and the costs keep their values, since a run that failed still spent them.
    """
    keys = ["algorithm", "scenario", "instance", "parameter", "value"]
    metrics = [c for c in wide.columns if c not in keys]

    long = wide.melt(id_vars=keys, value_vars=metrics, var_name="metric", value_name="result")

    # Blank the route measurements of failed runs. Each long row looks up the success of the run it came from
    success = wide.set_index(keys)["success"]
    failed = success.reindex(pd.MultiIndex.from_frame(long[keys])).values == 0
    long.loc[failed & long["metric"].isin(ROUTE_METRICS), "result"] = np.nan

    long = long.rename(columns={"algorithm": "planner"})

    return long[["planner", "scenario", "instance", "parameter", "value", "metric", "result"]].reset_index(drop=True)


def summarise_sweep(long):
    """
    Reduce a long sweep file to one row per (parameter, value, planner, metric): the median, the quartiles, how many maps went in and how many failed runs were left out. Pooled across the scenarios in the file.
    """
    rows = []

    for (parameter, value, planner, metric), block in long.groupby(["parameter", "value", "planner", "metric"], sort=False):
        results = block["result"]
        rows.append(
            {
                "parameter": parameter,
                "value": value,
                "planner": planner,
                "metric": metric,
                "median": round(float(results.median()), 4),
                "q25": round(float(results.quantile(0.25)), 4),
                "q75": round(float(results.quantile(0.75)), 4),
                "n_maps": len(block),
                "n_excluded": int(results.isna().sum()),
            }
        )

    return pd.DataFrame(rows)


def run_sweep(config, parameter, values, n_instances=6, algorithms=None, scenarios=None, workers=1, value_labels=None):
    """
    Sweep one parameter, write the long file and its chart, and return the path of the file.

    The file is sweep_<parameter>.csv in the configuration's output directory, and the chart is chart_sweep_<parameter>.png next to it. The maps and seeds are the same at every value, so a difference between two values is the parameter's doing.
    """
    from .plot import plot_sweep

    wide = sweep_parameter(
        config, parameter, values, algorithms=algorithms, n_instances=n_instances,
        scenarios=scenarios, workers=workers, value_labels=value_labels,
    )
    long = to_long(wide)

    csv_path = config.out_dir / f"sweep_{parameter}.csv"
    long.to_csv(csv_path, index=False)
    plot_sweep(long, parameter, SWEEP_METRICS, config.out_dir / f"chart_sweep_{parameter}.png")

    print(f"[sensitivity] saved {csv_path.name} and chart_sweep_{parameter}.png to {config.out_dir}")

    return csv_path


def ordering_holds(long, metric, better="high"):
    """
    Check whether the planners finish in the same order at every setting of the parameter, reading a long sweep file.

    This is the question a sensitivity analysis is really asking. The absolute numbers are expected to move when a parameter moves. What matters is whether the conclusion moves with them, and the conclusion is an ordering. Success is compared as a rate and everything else as a median over the runs that arrived.
    """
    block_of_metric = long[long["metric"] == metric]

    orderings = {}

    for value, block in block_of_metric.groupby("value"):
        if metric == "success":
            level = block.groupby("planner")["result"].mean()
        else:
            level = block.groupby("planner")["result"].median()

        # Rank so the best planner comes first whichever direction is better
        orderings[value] = tuple(
            level.sort_values(ascending=(better == "low")).index
        )

    # The ordering is stable when every setting produced the same one
    stable = len(set(orderings.values())) == 1

    return stable, orderings


def bootstrap_power(df, scenario, metric, baseline, other, sizes=POWER_SIZES, trials=500, alpha=0.05, seed=0):
    """
    Estimate how often a run of each size would detect the difference that was already measured.

    The paired rows from the completed run are resampled with replacement to make a run of the requested size, and the test the analysis would use is applied to it: McNemar on the paired outcomes for success, Wilcoxon on the paired differences for anything else. The share of resamples that come back significant is the power at that size.

    This is not a power calculation from an assumed effect. It reuses the variation that was actually observed, which is the honest way to answer how many maps the experiment needs, and it says so about the specific comparison it is given rather than about the experiment as a whole.

    500 resamples per size puts the estimated power within about two points of its true value, which is far finer than the decision it feeds. Raise trials if a number sits right on the target.

    The estimate inherits the limits of the data it is built from. On a twenty map pilot the observed effect is itself uncertain, so treat the answer as the order of magnitude of the sample size rather than a precise figure.
    """
    rng = np.random.default_rng(seed)

    # Line the two algorithms up map by map, which is what makes the differences paired. The success column is pulled in so route quality can be restricted to the maps both algorithms solved; when the metric is success itself it must not be asked for twice
    columns = ["instance", metric] if metric == "success" else ["instance", metric, "success"]

    block = df[df["scenario"] == scenario]
    left = block[block["algorithm"] == baseline][columns]
    right = block[block["algorithm"] == other][columns]
    merged = left.merge(right, on="instance", suffixes=("_base", "_other"))

    # Anything except success is only comparable on the maps both algorithms solved
    if metric != "success":
        merged = merged[(merged["success_base"] == 1) & (merged["success_other"] == 1)]

    base = merged[f"{metric}_base"].values.astype(float)
    other_values = merged[f"{metric}_other"].values.astype(float)
    differences = base - other_values

    # Nothing to resample from
    if len(differences) < 2:
        return pd.DataFrame()

    test = "mcnemar" if metric == "success" else "wilcoxon"
    rows = []

    for size in sizes:
        detected = 0

        for _ in range(trials):
            # Resample whole maps, so the pairing survives the resample
            picked = rng.integers(0, len(differences), size=size)

            if test == "mcnemar":
                p_value, _, _, _ = mcnemar(base[picked], other_values[picked])
                detected += int(not np.isnan(p_value) and p_value < alpha)
                continue

            sample = differences[picked]

            # A resample where every difference is identical has no rank information, so the test cannot run on it. All the same and non-zero is as clear a result as there is
            if np.allclose(sample, sample[0]):
                detected += int(abs(sample[0]) > 0)
                continue

            try:
                if wilcoxon(sample).pvalue < alpha:
                    detected += 1
            except ValueError:
                # SciPy refuses when every pair ties, which is a result rather than an error
                continue

        rows.append(
            {
                "scenario": scenario,
                "metric": metric,
                "comparison": f"{baseline} vs {other}",
                "test": test,
                "n_pairs_observed": len(differences),
                "n_instances": size,
                "power": round(detected / trials, 3),
                "observed_median_difference": round(float(np.median(differences)), 4),
                "observed_iqr": round(float(np.percentile(differences, 75) - np.percentile(differences, 25)), 4),
            }
        )

    return pd.DataFrame(rows)


def smallest_adequate_size(power_table, target=0.8):
    """Return the smallest size in the table that reaches the target power, or None when none of them do."""
    adequate = power_table[power_table["power"] >= target]

    return int(adequate["n_instances"].min()) if len(adequate) else None


def run_power_study(csv_path, out_dir, comparisons=POWER_COMPARISONS, target=0.8):
    """
    Work out how many maps each of the headline comparisons needs, save the table and return (power table, maps needed).

    The maps needed is the largest of the per-comparison answers, so the run is sized for the hardest claim it makes. None means one of the comparisons did not reach the target at the largest size tried.
    """
    out_dir = Path(out_dir)

    # The same collapsing the analysis does, so the pilot is read the way the real run will be
    df = collapse_repeats(load(csv_path))

    tables = []

    for scenario, metric, baseline, other in comparisons:
        table = bootstrap_power(df, scenario, metric, baseline, other)

        if len(table):
            tables.append(table)

    power = pd.concat(tables, ignore_index=True)
    power.to_csv(out_dir / "power_analysis.csv", index=False)

    print("\n============ how many maps each comparison needs ============")

    needed_sizes = []

    for comparison, block in power.groupby(["scenario", "metric", "comparison"], sort=False):
        needed = smallest_adequate_size(block, target)
        needed_sizes.append(needed)
        scenario, metric, pair = comparison
        answer = f"{needed} maps" if needed else f"more than {int(block['n_instances'].max())} maps"
        print(f"{scenario:13s} {metric:20s} {pair:20s} reaches {target:.0%} power at {answer}")

    # The run is sized for the hardest comparison. One that never reached the target leaves the answer open
    maps_needed = None if any(n is None for n in needed_sizes) else max(needed_sizes)
    if maps_needed is None:
        print(f"\n[sensitivity] at least one comparison does not reach {target:.0%} power by {max(POWER_SIZES)} maps")
    else:
        print(f"\n[sensitivity] every comparison reaches {target:.0%} power at {maps_needed} maps per scenario")

    print(f"[sensitivity] saved power_analysis.csv to {out_dir}")

    return power, maps_needed


def pilot_config(base):
    """The pilot's configuration: the base settings on the throwaway seed, twenty maps per scenario, one repeat per planner. One repeat makes the estimate conservative, since the real run pairs on the median of several."""
    settings = {
        field: getattr(base, field)
        for field in base.__dataclass_fields__
        if field != "out_dir"
    }
    settings.update(master_seed=PILOT_SEED, n_instances=PILOT_INSTANCES, n_repeats=1)

    return Config(out_dir=base.out_dir, **settings)


def run_pilot(base_config, out_dir, workers=1, target=0.8):
    """
    Run the pilot and size the experiment from it. Returns (power table, maps needed).

    The pilot is the core five planners plus the ablation cells on twenty maps per scenario on the throwaway seed. Its rows go to pilot_results.csv so the estimate can be rerun without the pilot, then the power study runs on them.
    """
    out_dir = Path(out_dir)
    config = pilot_config(base_config)

    print(f"[sensitivity] pilot on master seed {config.master_seed}, {config.n_instances} maps per scenario", flush=True)
    rows = StaticStudy(config, build_algorithms(ablation=True), workers=workers).run()
    csv_path = save_rows(rows, out_dir / "pilot_results.csv", STATIC_FIELDS)

    return run_power_study(csv_path, out_dir, target=target)


def tuning_config(base, **overrides):
    """The tuning configuration: the base settings on the tuning seed, twenty held-out maps per scenario, one repeat, plus whatever candidate values are being tried."""
    settings = {
        field: getattr(base, field)
        for field in base.__dataclass_fields__
        if field != "out_dir"
    }
    settings.update(master_seed=TUNING_SEED, n_instances=TUNING_INSTANCES, n_repeats=1)
    settings.update(overrides)

    return Config(out_dir=base.out_dir, **settings)


def choose_setting(rows, parameter, default):
    """
    Pick one value of a parameter from the rows its candidates produced. Returns (chosen value, table of candidates).

    Success comes first: the candidate that solves the most maps wins, and a candidate within one map of the best is still in the running. Among those, the lowest median drive time when slowing for the turns wins, since that is the number the project is about. Planning time is left out of the choice because it is not comparable across worker processes and the budgets are bounded by the counters anyway. A tie keeps the default, so nothing moves without a reason.
    """
    block = rows[rows["parameter"] == parameter]

    table = []
    for value, candidate in block.groupby("value"):
        solved = candidate[candidate["success"] == 1]
        table.append(
            {
                "parameter": parameter,
                "value": value,
                "n_maps": len(candidate),
                "n_solved": int(candidate["success"].sum()),
                "success": round(float(candidate["success"].mean()), 4),
                "drive_time_curved": round(float(solved["drive_time_curved"].median()), 4) if len(solved) else float("inf"),
            }
        )
    table = pd.DataFrame(table)

    # Everything within one map of the best is a contender. Counted in maps rather than in rounded rates, so a candidate one map short is never pushed out by the rounding
    best = table["n_solved"].max()
    contenders = table[table["n_solved"] >= best - 1]

    # The fastest drive among the contenders, the default winning any tie
    fastest = contenders["drive_time_curved"].min()
    winners = contenders[contenders["drive_time_curved"] <= fastest + 1e-9]
    if default in set(winners["value"]):
        chosen = default
    else:
        chosen = winners["value"].iloc[0]

    table["chosen"] = table["value"] == chosen

    return chosen, table


def tune_planner(name, base_config, grid, workers=1):
    """
    Tune one planner over its grid, one parameter at a time around the base settings. Returns (chosen values, the rows every candidate produced).

    The base setting is run once and shared by every parameter as its default candidate, so the budget is the default plus every other value. Each candidate runs on the tuning maps with the full harness, workers included, and the rows carry the planner, the parameter and the value.
    """
    spec = {s.name: s for s in build_algorithms(ablation=True, smoothers=True)}[name]

    def run(**overrides):
        config = tuning_config(base_config, **overrides)
        rows = StaticStudy(config, [spec], workers=workers).run()
        return pd.DataFrame(rows)

    print(f"[tuning] {name}: default setting", flush=True)
    default_rows = run()

    frames = []
    chosen = {}

    for parameter, values in grid.items():
        default = getattr(base_config, parameter)

        # The shared default rows, labelled for this parameter
        frames.append(default_rows.assign(planner=name, parameter=parameter, value=default))

        for value in values:
            if value == default:
                continue
            print(f"[tuning] {name}: {parameter} = {value}", flush=True)
            frames.append(run(**{parameter: value}).assign(planner=name, parameter=parameter, value=value))

    rows = pd.concat(frames, ignore_index=True)

    for parameter in grid:
        chosen[parameter], _ = choose_setting(rows, parameter, getattr(base_config, parameter))

    return chosen, rows


def run_tuning(base_config, out_dir, workers=1):
    """
    Tune every planner in order and save what was tried and what was chosen. Returns the chosen values as one dictionary.

    Each planner is tuned with the values chosen for the planners before it already applied, so the hybrids are tuned on top of the tuned APF and RRT*. The chosen values are printed for locking into config.py by hand, with the seed and the date, since a value that changes on its own is a value nobody can account for.
    """
    out_dir = Path(out_dir)
    config = base_config
    all_rows = []
    all_choices = []
    chosen = {}

    for name in TUNING_ORDER:
        planner_choice, rows = tune_planner(name, config, TUNING_GRID[name], workers=workers)
        all_rows.append(rows)

        for parameter, value in planner_choice.items():
            _, table = choose_setting(rows, parameter, getattr(config, parameter))
            all_choices.append(table.assign(planner=name))
            chosen[parameter] = value

        # The next planner sees this one's choices
        config = _paired_config(config, **planner_choice)

    pd.concat(all_rows, ignore_index=True).to_csv(out_dir / "tuning_results.csv", index=False)
    choices = pd.concat(all_choices, ignore_index=True)
    choices.to_csv(out_dir / "tuning_choices.csv", index=False)

    print("\n============ tuning: candidates and choices ============")
    print(choices.to_string(index=False))
    print(f"\n[tuning] seed {TUNING_SEED}, {TUNING_INSTANCES} maps per scenario, budget {sum(len(v) - 1 for v in TUNING_GRID['APF'].values()) + 1} settings per planner")
    print("[tuning] chosen: " + ", ".join(f"{k} = {v}" for k, v in chosen.items()))
    print(f"[tuning] saved tuning_results.csv and tuning_choices.csv to {out_dir}")

    return chosen


def run_sensitivity_study(config, parameters=None, n_instances=6, scenarios=None, workers=1):
    """
    Sweep each parameter in turn, each to its own long file and chart, and save both the per-setting summary and a verdict on whether the ordering of the planners survived.
    """
    parameters = parameters or list(SWEEPS)

    summaries = []
    verdicts = []

    for parameter in parameters:
        print(f"=== sweeping {parameter} ===", flush=True)

        long = pd.read_csv(run_sweep(
            config, parameter, SWEEPS[parameter], n_instances=n_instances, scenarios=scenarios, workers=workers
        ))
        summaries.append(summarise_sweep(long))

        for metric, better in (("success", "high"), ("drive_time_curved", "low")):
            stable, orderings = ordering_holds(long, metric, better)

            verdicts.append(
                {
                    "parameter": parameter,
                    "metric": metric,
                    "ordering_stable": stable,
                    "n_orderings": len(set(orderings.values())),
                    "best_at_each_setting": ", ".join(
                        f"{value}:{order[0]}" for value, order in orderings.items()
                    ),
                }
            )

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(config.out_dir / "sensitivity_summary.csv", index=False)

    verdict = pd.DataFrame(verdicts)
    verdict.to_csv(config.out_dir / "sensitivity_verdict.csv", index=False)

    print("\n============ does the ordering survive the parameters moving ============")
    print(verdict.to_string(index=False))
    print(f"\n[sensitivity] saved sensitivity_summary.csv and sensitivity_verdict.csv to {config.out_dir}")

    return summary, verdict


def main():
    """
    Command line entry point, so both studies can be run without writing a script.

        python -m planning.experiments.sensitivity --tune --workers 4
        python -m planning.experiments.sensitivity --pilot --workers 4
        python -m planning.experiments.sensitivity --power outputs/pilot_results.csv
        python -m planning.experiments.sensitivity --sweep
        python -m planning.experiments.sensitivity --goal-offset --instances 20 --workers 4
        python -m planning.experiments.sensitivity --planner-sweep APF --instances 30 --workers 4 --out results/sweeps
        python -m planning.experiments.sensitivity --plot results/sweeps/sweep_apf_step.csv
        python -m planning.experiments.sensitivity --sweep --parameter apf_step --instances 8
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Check the experiment itself: how far the results move with the parameters, and how many maps they need."
    )
    parser.add_argument(
        "--power",
        metavar="CSV",
        help="estimate the maps needed for each headline comparison, from a results file already collected",
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help=f"run {PILOT_INSTANCES} maps per scenario on the throwaway seed {PILOT_SEED} and size the experiment from them",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help=f"tune every planner one parameter at a time on {TUNING_INSTANCES} held-out maps per scenario from seed {TUNING_SEED}",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="processes to spread a pilot's, a tuning's or a sweep's maps across"
    )
    parser.add_argument(
        "--sweep", action="store_true", help="vary each parameter in turn and rerun the comparison"
    )
    parser.add_argument(
        "--planner-sweep", metavar="PLANNER", choices=list(PLANNER_SWEEPS),
        help="sweep every parameter of one planner over its grid, thirty maps per value on every scenario, one long file and chart per parameter",
    )
    parser.add_argument(
        "--plot", metavar="CSV", nargs="+",
        help="redraw the chart for each of these long sweep files without rerunning anything",
    )
    parser.add_argument(
        "--out", default=None, help="where the sweep files go; the configuration's output directory otherwise",
    )
    parser.add_argument(
        "--goal-offset", action="store_true",
        help=f"move the goal off the diagonal by each of {GOAL_OFFSETS} cells and draw how the planners respond",
    )
    parser.add_argument(
        "--parameter",
        choices=list(SWEEPS),
        help="sweep only this parameter instead of all of them",
    )
    parser.add_argument(
        "--instances", type=int, default=6, help="maps per setting inside the sweep"
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.8,
        help="the power a sample size has to reach before it counts as adequate",
    )
    arguments = parser.parse_args()

    # Nothing asked for is almost certainly a mistake rather than a request to do nothing
    if not any((arguments.power, arguments.sweep, arguments.pilot, arguments.tune, arguments.goal_offset, arguments.planner_sweep, arguments.plot)):
        parser.error("choose --tune, --pilot, --power, --sweep, --planner-sweep, --plot or --goal-offset.")

    config = Config(out_dir=arguments.out) if arguments.out else Config()

    if arguments.planner_sweep:
        run_planner_sweeps(arguments.planner_sweep, config, n_instances=arguments.instances, workers=arguments.workers)

    if arguments.plot:
        from .plot import plot_sweep

        for csv_path in arguments.plot:
            long = pd.read_csv(csv_path)
            parameter = long["parameter"].iloc[0]
            plot_sweep(long, parameter, SWEEP_METRICS, Path(csv_path).with_name(f"chart_sweep_{parameter}.png"))

    if arguments.tune:
        run_tuning(config, config.out_dir, workers=arguments.workers)

    if arguments.pilot:
        run_pilot(config, config.out_dir, workers=arguments.workers, target=arguments.target)

    if arguments.power:
        run_power_study(arguments.power, config.out_dir, target=arguments.target)

    if arguments.sweep:
        parameters = [arguments.parameter] if arguments.parameter else None
        run_sensitivity_study(config, parameters=parameters, n_instances=arguments.instances, workers=arguments.workers)

    if arguments.goal_offset:
        from .plot import plot_goal_offset

        rows = sweep_goal_offset(config, n_instances=arguments.instances, workers=arguments.workers)
        rows.to_csv(config.out_dir / "goal_offset_sweep.csv", index=False)
        plot_goal_offset(rows, config.out_dir / "chart_goal_offset.png")
        print(f"[sensitivity] saved goal_offset_sweep.csv and chart_goal_offset.png to {config.out_dir}")


if __name__ == "__main__":
    main()
