"""
experiments/analyse.py

Reads the raw results and produces the two tables the results chapter is written from.

Every table is paired on the map. A stochastic planner is run on several seeds per map, and its repeats are collapsed to one row per map first: the per-map median is the paired value and the within-map spread goes to a reliability table of its own.

The summary table gives the median and the interquartile range of every measurement for every combination of scenario and algorithm. Planning times and route lengths are skewed, a mean of them is pulled about by the odd bad map, and the rank test used below is a test on medians, so the table reports what the test tests. The rates are the exception: a rate is a mean by definition and carries a Wilson interval instead of a spread.

The significance tests compare one chosen algorithm against each of the others. Because every algorithm ran on the same maps, the two can be compared map by map, which is a paired comparison and needs far fewer instances than comparing two independent groups would. The Wilcoxon signed-rank test is used for the measured quantities because it makes no assumption that the data is bell shaped, and path lengths and computation times usually are not.

Success is binary, so it gets McNemar's test on the paired outcomes instead: only the maps where the two algorithms disagreed carry any information, and the test asks whether the disagreements lean one way. Under twenty five discordant maps the exact binomial form is used, since the chi-square approximation is poor on small counts. Every success rate is reported with a 95% Wilson interval and the number of maps it came from, because a rate on its own hides how little thirty maps can pin it down.

A p-value only says whether a difference is there. The rank-biserial correlation is reported alongside the Wilcoxon tests to say how large the difference is, since with fifty paired instances a difference of no practical size can still come out significant. It does not apply to a binary outcome; the McNemar rows report the two discordant counts instead.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, norm, wilcoxon

from ..utils.config import Config
from ..utils.metrics import mission_time

# The measurements summarised for every scenario and algorithm
METRICS = [
    "success",
    "path_length",
    "length_ratio",
    "comp_time",
    "mission_time",
    "mission_time_curved",
    "smoothness",
    "curvature",
    "max_curvature",
    "clearance_min",
    "clearance_mean",
    "total_turning",
    "sharp_turns",
    "radius_violated",
    "switches",
    "false_alarm",
    "time_to_detection",
    "collision_checks",
]

# The measurements that are rates. These are reported as means because a rate is a mean, and they get Wilson intervals rather than a spread
RATES = ("success", "radius_violated", "false_alarm")

# The measured quantities the Wilcoxon test is run on, which are the ones the claims of the project rest on. Success is not among them: it is binary and goes to McNemar
WILCOXON_METRICS = ["mission_time", "max_curvature"]

# Everything the significance table covers
TESTED_METRICS = ["success"] + WILCOXON_METRICS

# How many discordant maps McNemar needs before the chi-square form is trusted over the exact one
MCNEMAR_EXACT_BELOW = 25

# The four cells of the ablation: which entry has adaptive stuck detection on and which has smoothing on
ABLATION_CELLS = {
    "Hybrid": (False, False),
    "Adaptive": (True, False),
    "Bezier": (False, True),
    "Modified": (True, True),
}

# The measurements the ablation is read on
ABLATION_METRICS = ("success", "length_ratio", "mission_time_curved", "max_curvature", "sharp_turns")

# The three comparisons the conclusions rest on, fixed before the results were looked at: the modified hybrid against the standard hybrid, against APF and against RRT*. Every other pair in the table is descriptive, which means it is there to be read and not to be claimed
CONFIRMATORY = (
    ("Modified", "Hybrid"),
    ("Modified", "APF"),
    ("Modified", "RRT*"),
)


def load(csv_path, speed=None):
    """Read a results CSV into a pandas table, with the mission time worked out from the columns already in it."""
    return add_mission_time(pd.read_csv(csv_path), speed)


def add_mission_time(df, speed=None):
    """
    Add the mission time column: the planning time plus the time spent driving the route at the assumed speed.

    It is derived rather than recorded because both parts of it are already in the results file. That also means a results file written before this column existed gains it as soon as it is read, with no need to run the experiment again.
    """
    # The speed comes from the configuration unless the caller wants to see what a different one would do
    if speed is None:
        speed = Config.vehicle_speed

    # A file without both parts cannot have a mission time, which is the case for the dynamic results
    if not {"path_length", "comp_time"}.issubset(df.columns):
        return df

    df = df.copy()
    df["mission_time"] = [
        round(mission_time(distance, planning, speed), 4)
        for distance, planning in zip(df["path_length"], df["comp_time"])
    ]

    # The second mission time slows for the turns. Its driving half is recorded per run since it needs the route, and only the planning time is added here. A file written before that column existed keeps to the constant speed number
    if "drive_time_curved" in df.columns:
        df["mission_time_curved"] = [
            round(planning + driving, 4)
            for planning, driving in zip(df["comp_time"], df["drive_time_curved"])
        ]

    return df


def collapse_repeats(df):
    """
    Reduce the repeats of one planner on one map to a single row, which is the paired value every comparison is made on.

    The value of a route column is the median over the successful repeats, so a repeat that failed does not drag the map's length towards zero. Success is the majority of the repeats, which keeps it binary for McNemar. Planning time, escapes and check counts are medians over every repeat since a failed run spent them too. n_repeats says how many runs the row stands for. A file without a repeat column, or with only repeat 0, comes back as it was.
    """
    if "repeat" not in df.columns or df["repeat"].nunique() <= 1:
        collapsed = df.drop(columns=["repeat"], errors="ignore").copy()
        collapsed["n_repeats"] = 1
        return collapsed

    keys = ["scenario", "algorithm", "instance"]

    # Route columns are blanked on failed repeats first, so the median below is over the successes
    masked = mask_failures(df).drop(columns=["repeat"])
    numeric = [c for c in masked.columns if c not in keys and pd.api.types.is_numeric_dtype(masked[c])]

    grouped = masked.groupby(keys, sort=False)
    collapsed = grouped[numeric].median()

    # Majority vote, with an even split counted as a failure since half the runs did not arrive
    collapsed["success"] = (grouped["success"].mean() > 0.5).astype(int)
    collapsed["n_repeats"] = grouped.size()

    return collapsed.reset_index()


def reliability_table(df, metrics=("success", "length_ratio", "path_length", "mission_time_curved", "max_curvature")):
    """
    How much a planner's answer on one map depends on its seed, per scenario and algorithm.

    For success it is the share of maps where the repeats disagreed with each other. For a route column it is the median over maps of the within-map interquartile range across the successful repeats; a map with fewer than two successful repeats has no spread to measure and is left out of that median. Only planners with more than one repeat appear, since a single run has no spread to report.
    """
    if "repeat" not in df.columns:
        return pd.DataFrame()

    keys = ["scenario", "algorithm", "instance"]
    masked = mask_failures(df)
    rows = []

    for (scenario, algorithm), block in masked.groupby(["scenario", "algorithm"], sort=False):
        per_map = block.groupby("instance")
        repeats = int(per_map.size().max())
        if repeats <= 1:
            continue

        row = {"scenario": scenario, "algorithm": algorithm, "n_repeats": repeats, "n_maps": per_map.ngroups}

        for metric in metrics:
            if metric not in block.columns:
                continue
            if metric == "success":
                row["success_disagreement"] = round(float((per_map["success"].nunique() > 1).mean()), 4)
            else:
                row[f"{metric}_within_iqr"] = round(float(per_map[metric].agg(_spread).median()), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def _spread(values):
    """The interquartile range of the repeats on one map, or nan when fewer than two of them have a value."""
    if values.notna().sum() < 2:
        return np.nan
    return iqr(values)


def mask_failures(df):
    """
    Return a copy of the table with the route-quality columns of failed runs blanked out.

    A run that never arrived has a path length of zero in the raw file. Averaging that against the runs that did arrive would reward failing, since the algorithm that gives up soonest would post the shortest routes. Blanking the value leaves those runs out of the length, smoothness and clearance averages while they still count against the success rate.

    Computation time, the number of escapes and the check count keep their values, because a run that failed still spent that time, made those switches and asked about those cells.
    """
    masked = df.copy()

    # The columns that only mean something for a run that arrived
    quality = [
        "path_length",
        "length_ratio",
        "mission_time",
        "mission_time_curved",
        "drive_time_curved",
        "smoothness",
        "curvature",
        "max_curvature",
        "clearance_min",
        "clearance_mean",
        "total_turning",
        "sharp_turns",
        "radius_violated",
    ]

    masked.loc[masked["success"] == 0, quality] = np.nan

    return masked


def iqr(values):
    """The interquartile range, the spread of the middle half. It is the spread that goes with a median."""
    values = values.dropna()
    if len(values) == 0:
        return np.nan
    return float(values.quantile(0.75) - values.quantile(0.25))


def summary_table(df):
    """
    Median and interquartile range of every measurement, grouped by scenario and algorithm, with the rates as means. The route-quality columns cover the successful runs only.
    """
    # Which statistics each column gets
    statistics = {}
    for metric in METRICS:
        statistics[metric] = ["mean"] if metric in RATES else ["median", iqr]

    return (
        mask_failures(df)
        .groupby(["scenario", "algorithm"])
        .agg(statistics)
        .round(3)
    )


def wilson_interval(successes, n, confidence=0.95):
    """
    The Wilson score interval for a rate of successes out of n, as (low, high). Unlike the textbook normal interval it stays inside 0 and 1 and still gives a usable upper bound when nothing succeeded.
    """
    if n == 0:
        return float("nan"), float("nan")

    z = norm.ppf(1 - (1 - confidence) / 2)
    rate = successes / n

    # The interval is centred a little towards a half and shrinks with n, which is what keeps it honest on small counts
    denominator = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denominator
    half_width = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator

    return float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))


def mcnemar(base, other):
    """
    McNemar's test on two paired columns of 0 and 1. Returns (p_value, base_only, other_only, form): the p-value, how many maps only the baseline solved, how many only the other solved, and which form of the test gave the answer.

    Maps both solved or both failed say nothing about which is better, so only the discordant ones are counted. Under MCNEMAR_EXACT_BELOW of them the exact two-sided binomial test is used; above that the chi-square with the continuity correction.
    """
    base = np.asarray(base, dtype=int)
    other = np.asarray(other, dtype=int)

    base_only = int(np.sum((base == 1) & (other == 0)))
    other_only = int(np.sum((base == 0) & (other == 1)))
    discordant = base_only + other_only

    # No disagreement at all is no evidence either way
    if discordant == 0:
        return float("nan"), base_only, other_only, "mcnemar (no discordant pairs)"

    if discordant < MCNEMAR_EXACT_BELOW:
        p_value = binomtest(min(base_only, other_only), discordant, 0.5).pvalue
        return float(p_value), base_only, other_only, "mcnemar exact"

    statistic = (abs(base_only - other_only) - 1) ** 2 / discordant
    return float(chi2.sf(statistic, 1)), base_only, other_only, "mcnemar chi2"


def success_rates(df, group=("scenario", "algorithm"), column="success"):
    """
    One row per group with the rate, its 95% Wilson interval and the number of maps it came from. This is the table any success rate is quoted from, so no rate goes out without its interval and its count.
    """
    rows = []

    for keys, block in df.groupby(list(group)):
        if not isinstance(keys, tuple):
            keys = (keys,)

        n = len(block)
        successes = int(block[column].sum())
        low, high = wilson_interval(successes, n)

        row = dict(zip(group, keys))
        row.update(
            {
                "n_maps": n,
                column: round(successes / n, 4) if n else np.nan,
                f"{column}_low": round(low, 4),
                f"{column}_high": round(high, 4),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def rank_biserial(x, y):
    """
    A paired effect size between -1 and 1. It is the share of pairs where x is larger minus the share where y is larger, so a value near zero means the two swap places about equally often and a value near 1 means one of them wins almost every pair.
    """
    # Both as numpy arrays of type float so the comparison is elementwise
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # How many pairs each side wins
    difference = x - y
    wins = int(np.sum(difference > 0))
    losses = int(np.sum(difference < 0))

    # Pairs that tie carry no information about which is better, so they are left out of the denominator
    decided = wins + losses
    if decided == 0:
        return 0.0

    return float((wins - losses) / decided)


def paired_tests(df, metric, baseline, scenario=None):
    """
    Compare the baseline algorithm against every other algorithm on one measurement, and return a table with the two means, the p-value and the effect size.

    Success goes to McNemar on the paired outcomes and reports the two discordant counts in place of an effect size. Everything else goes to Wilcoxon, and only the instances that both algorithms solved are compared: the length of a route that was never found is not a number that can be compared against one that was. That makes every quality comparison a comparison over the easier maps, so the table carries the number of maps that survived next to the number there were.

    Each row is labelled confirmatory when it is one of the three comparisons fixed in advance, and descriptive otherwise.
    """
    # Restrict to one scenario when asked, since the algorithms behave very differently across them and pooling would hide that
    if scenario is not None:
        df = df[df["scenario"] == scenario]

    rows = []

    for other in [a for a in df["algorithm"].unique() if a != baseline]:
        # Line the two algorithms up map by map on the instance number. Success is pulled in so the route columns can be restricted to solved maps; when success is itself the metric it must not be asked for twice, or the frame gets two columns of one name and every count doubles
        columns = ["instance", metric] if metric == "success" else ["instance", metric, "success"]
        base = df[df["algorithm"] == baseline][columns]
        comparison = df[df["algorithm"] == other][columns]
        merged = base.merge(comparison, on="instance", suffixes=("_base", "_other"))

        # How many maps the two algorithms met, before the survivors are picked out
        n_maps = len(merged)

        # Keep only the instances both algorithms solved, except when the measurement being compared is success itself
        if metric != "success":
            merged = merged[
                (merged["success_base"] == 1) & (merged["success_other"] == 1)
            ]

        x = merged[f"{metric}_base"].values
        y = merged[f"{metric}_other"].values

        base_only = np.nan
        other_only = np.nan

        if metric == "success":
            p_value, base_only, other_only, test = mcnemar(x, y)
            effect = float("nan")
        elif len(x) < 3 or np.allclose(x, y):
            # Too few pairs, or two identical columns, leaves the test with nothing to work on
            p_value = float("nan")
            effect = 0.0
            test = "wilcoxon"
        else:
            try:
                p_value = float(np.ravel(wilcoxon(x, y).pvalue)[0])
            except (ValueError, ZeroDivisionError):
                # SciPy raises when every pair ties, which is a result rather than an error
                p_value = float("nan")
            effect = rank_biserial(x, y)
            test = "wilcoxon"

        rows.append(
            {
                "metric": metric,
                "kind": "confirmatory" if (baseline, other) in CONFIRMATORY else "descriptive",
                "test": test,
                "baseline": baseline,
                "vs": other,
                "n_maps": n_maps,
                "n_pairs": len(x),
                "baseline_median": round(float(np.median(x)), 3) if len(x) else np.nan,
                "other_median": round(float(np.median(y)), 3) if len(y) else np.nan,
                "p_value": round(p_value, 4) if not np.isnan(p_value) else np.nan,
                "effect_size": round(effect, 3) if not np.isnan(effect) else np.nan,
                "base_only": base_only,
                "other_only": other_only,
            }
        )

    return pd.DataFrame(rows)


def ablation_table(df, metrics=ABLATION_METRICS):
    """
    Read the two extensions as a 2 x 2 factorial rather than as three arms against a baseline.

    The four cells are the standard hybrid, adaptive detection alone, smoothing alone and both together. For each map the main effect of an extension is the average of the two cells that have it minus the average of the two that do not, and the interaction is half the difference between what smoothing adds with adaptive detection on and what it adds with it off. Those per-map effects are summarised by their median, and a one-sample Wilcoxon test on them asks whether the effect is reliably on one side of zero. Success is a rate, so its effects are differences of rates over every map; every other measurement uses only the maps all four cells solved, and n_pairs says how many that was.

    Returns one row per scenario and metric, or an empty table when the four cells are not all in the results.
    """
    if not set(ABLATION_CELLS).issubset(df["algorithm"].unique()):
        return pd.DataFrame()

    rows = []

    for scenario, block in df.groupby("scenario"):
        cells = {}
        for name in ABLATION_CELLS:
            cells[name] = block[block["algorithm"] == name].set_index("instance")

        # Every map the four cells all met
        instances = sorted(set.intersection(*(set(cell.index) for cell in cells.values())))

        for metric in metrics:
            # Success counts every map; the quality columns count only the maps all four cells solved
            if metric == "success":
                kept = instances
            else:
                kept = [i for i in instances if all(cells[name].loc[i, "success"] == 1 for name in cells)]

            if not kept:
                continue

            value = {name: cells[name].loc[kept, metric].astype(float).values for name in cells}

            adaptive = (value["Adaptive"] + value["Modified"]) / 2 - (value["Hybrid"] + value["Bezier"]) / 2
            smoothing = (value["Bezier"] + value["Modified"]) / 2 - (value["Hybrid"] + value["Adaptive"]) / 2
            interaction = ((value["Modified"] - value["Bezier"]) - (value["Adaptive"] - value["Hybrid"])) / 2

            row = {
                "scenario": scenario,
                "metric": metric,
                "n_maps": len(instances),
                "n_pairs": len(kept),
            }

            # The cells themselves, as rates for success and medians for the rest
            summarise = np.mean if metric == "success" else np.median
            row["cell_fixed_none"] = round(float(summarise(value["Hybrid"])), 4)
            row["cell_adaptive_none"] = round(float(summarise(value["Adaptive"])), 4)
            row["cell_fixed_bezier"] = round(float(summarise(value["Bezier"])), 4)
            row["cell_adaptive_bezier"] = round(float(summarise(value["Modified"])), 4)

            for label, effect in (("adaptive", adaptive), ("smoothing", smoothing), ("interaction", interaction)):
                key = "interaction" if label == "interaction" else f"effect_{label}"
                row[key] = round(float(summarise(effect)), 4)
                row[f"{key}_p"] = _one_sample_p(effect)

            rows.append(row)

    return pd.DataFrame(rows)


def _one_sample_p(differences):
    """Wilcoxon signed-rank p-value for the differences sitting on one side of zero, or nan when there is nothing to test."""
    differences = np.asarray(differences, dtype=float)
    if len(differences) < 3 or np.allclose(differences, 0.0):
        return np.nan
    try:
        return round(float(wilcoxon(differences).pvalue), 4)
    except ValueError:
        return np.nan


def run_full_analysis(csv_path, out_dir, baseline="Modified"):
    """
    Build both tables from the static results, save them and print a short version to the terminal. Returns (summary, tests).
    """
    out_dir = Path(out_dir)
    raw = load(csv_path)

    # Every table below is paired on the map, so the repeats of a planner on a map become one row first. The spread they had is reported on its own
    df = collapse_repeats(raw)

    reliability = reliability_table(raw)
    if len(reliability):
        reliability.to_csv(out_dir / "reliability.csv", index=False)
        print("\n============ reliability: how much one map's answer depends on the seed ============")
        print(reliability.to_string(index=False))

    ### === The summary table === ###
    summary = summary_table(df)
    summary.to_csv(out_dir / "summary_table.csv")

    print("\n==================== summary (success rate, medians) ====================")

    # The full table with its spreads is saved, but only a few columns are printed so the terminal output stays readable
    shown = ["success", "length_ratio", "comp_time", "mission_time", "max_curvature", "clearance_min"]
    print(summary[[(metric, "mean" if metric in RATES else "median") for metric in shown]])
    ### === END ===

    ### === The success rates with their intervals === ###
    rates = success_rates(df)
    rates.to_csv(out_dir / "success_rates.csv", index=False)

    print("\n==================== success rates (95% Wilson) ====================")
    print(rates.to_string(index=False))
    ### === END ===

    ### === The significance tests === ###
    all_tests = []

    for scenario in df["scenario"].unique():
        for metric in TESTED_METRICS:
            test = paired_tests(df, metric=metric, baseline=baseline, scenario=scenario)
            test.insert(0, "scenario", scenario)
            all_tests.append(test)

    tests = pd.concat(all_tests, ignore_index=True)
    tests.to_csv(out_dir / "significance_tests.csv", index=False)

    print(f"\n============ significance ({baseline} against the others) ============")
    print(tests.to_string(index=False))
    ### === END ===

    ### === The stuck detector: false alarms and time to detection === ###
    normal = df[(df["scenario"] == "normal_city") & df["false_alarm"].notna()]
    if len(normal):
        alarms = success_rates(normal, group=("algorithm",), column="false_alarm")
        alarms.to_csv(out_dir / "false_alarms.csv", index=False)
        print("\n============ false alarms on the normal city (share of maps with an escape, 95% Wilson) ============")
        print(alarms.to_string(index=False))

    detected = df[(df["scenario"] == "blocked_road") & df["time_to_detection"].notna()]
    if len(detected):
        detection = detected.groupby("algorithm")["time_to_detection"].agg(["count", "median", iqr]).round(1)
        detection.to_csv(out_dir / "time_to_detection.csv")
        print("\n============ time to detection on the blocked road (path points from pocket entry to first escape) ============")
        print(detection.to_string())
    ### === END ===

    ### === The ablation as a factorial === ###
    ablation = ablation_table(df)

    if len(ablation):
        ablation.to_csv(out_dir / "ablation_table.csv", index=False)
        print("\n============ ablation, 2 x 2: adaptive detection by smoothing ============")
        print(ablation.to_string(index=False))
    ### === END ===

    print(f"\n[analyse] saved summary_table.csv, success_rates.csv, significance_tests.csv and ablation_table.csv to {out_dir}")

    return summary, tests


def summarise_dynamic(csv_path, out_dir):
    """
    Summarise the dynamic results, where what matters is how often each way of driving arrived, how often it was hit and what it cost in planning time and replans.
    """
    out_dir = Path(out_dir)
    df = load(csv_path)

    # One block per obstacle speed and driver. The three rates are means with Wilson intervals; everything else is a median with its spread, as in the static table
    keys = ["speed_factor", "driver"] if "speed_factor" in df.columns else ["driver"]
    by_driver = df.groupby(keys)
    summary = by_driver[["success", "reached", "collided"]].mean().round(3)

    for column in ("success", "reached", "collided"):
        rates = success_rates(df, group=tuple(keys), column=column).set_index(keys)
        summary[f"{column}_low"] = rates[f"{column}_low"]
        summary[f"{column}_high"] = rates[f"{column}_high"]
    summary["n_maps"] = by_driver.size()

    for column in ("steps", "replans", "comp_time", "collision_checks", "path_length"):
        summary[f"{column}_median"] = by_driver[column].median().round(3)
        summary[f"{column}_iqr"] = by_driver[column].agg(iqr).round(3)

    summary.to_csv(out_dir / "dynamic_summary.csv")

    print("\n============ dynamic scenario (rates and medians per obstacle speed and driver) ============")
    print(summary[["success", "success_low", "success_high", "collided", "replans_median", "n_maps"]])
    print(f"\n[analyse] saved dynamic_summary.csv to {out_dir}")

    return summary
