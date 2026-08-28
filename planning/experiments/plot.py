"""
experiments/plot.py

Turns the results CSV into the bar charts for the results chapter, one chart per measurement, with the scenarios along the x axis and one bar per algorithm inside each scenario.

Only matplotlib is used so the project keeps its dependency list short.

Each bar is the median across the instances and the error bar runs from the lower to the upper quartile, so it describes how much the algorithm varies from map to map rather than how precisely a mean is known, and it is not pulled about by the one bad map the way a mean and a standard deviation are. The rates, success among them, are drawn as the rate with no bar: their uncertainty is a Wilson interval in the tables.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyse import RATES, collapse_repeats, load, mask_failures, wilson_interval

# A fixed colour per algorithm so an algorithm looks the same on every chart
COLOURS = {
    "APF": "#b3402f",
    "RRT*": "#2f6fb3",
    "RRT*+Bezier": "#8fb3dc",
    "Hybrid": "#6b7280",
    "Modified": "#e9a13b",
    "Adaptive": "#2a9d8f",
    "Bezier": "#7a3b8a",
    "BSpline": "#4f7942",
    "Dubins": "#8a5e12",
}

# The charts to draw, as (column, axis label, title, whether a larger value is better)
CHARTS = [
    ("success", "success rate", "Success rate", True),
    ("length_ratio", "path length over the grid optimum", "Path length ratio", False),
    ("comp_time", "mean computation time (s)", "Computation time", False),
    ("mission_time", "planning plus driving (s)", "Total mission time", False),
    ("mission_time_curved", "planning plus driving, slowing for turns (s)", "Mission time with turns", False),
    ("smoothness", "smoothness (0 to 1)", "Path smoothness per vertex", False),
    ("curvature", "radians turned per cell", "Path curvature", False),
    ("max_curvature", "radians per cell at the tightest turn", "Tightest turn on the route", False),
    ("clearance_min", "closest approach to an obstacle (cells)", "Minimum clearance", True),
    ("clearance_mean", "mean clearance (cells)", "Obstacle clearance", True),
]


def bar_chart(df, metric, ylabel, title, save_path, higher_is_better=True):
    """
    Draw one grouped bar chart, where each group is a scenario and each bar in the group is an algorithm, and write it to save_path.
    """
    # The groups along the x axis and the bars inside each group
    scenarios = list(df["scenario"].unique())
    algorithms = list(df["algorithm"].unique())

    # The height of every bar and the two ends of its error bar, looked up by (scenario, algorithm). A rate is drawn as the rate with no spread; anything else is the median with the quartiles
    grouped = df.groupby(["scenario", "algorithm"])[metric]
    if metric in RATES:
        centres = grouped.mean()
        lower = centres
        upper = centres
    else:
        centres = grouped.median()
        lower = grouped.quantile(0.25)
        upper = grouped.quantile(0.75)

    # One position on the x axis per scenario, with the bars sharing the space between positions
    positions = np.arange(len(scenarios))
    width = 0.8 / len(algorithms)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, algorithm in enumerate(algorithms):
        # An algorithm missing from a scenario leaves a gap rather than a bar of height zero, which would read as a real result
        heights = [centres.get((scenario, algorithm), np.nan) for scenario in scenarios]
        below = [h - lower.get((scenario, algorithm), h) for scenario, h in zip(scenarios, heights)]
        above = [upper.get((scenario, algorithm), h) - h for scenario, h in zip(scenarios, heights)]
        errors = np.nan_to_num(np.array([below, above]))

        # Shift this algorithm's bars off the group centre so the bars sit side by side
        offset = (i - (len(algorithms) - 1) / 2) * width

        ax.bar(
            positions + offset,
            heights,
            width,
            yerr=errors,
            capsize=3,
            label=algorithm,
            color=COLOURS.get(algorithm, "#333333"),
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(ylabel)

    # Say which direction is better on the chart itself, since it changes between measurements
    direction = "higher is better" if higher_is_better else "lower is better"
    ax.set_title(f"{title} ({direction})", fontsize=12)

    ax.legend(fontsize=8, ncol=len(algorithms))

    # A faint horizontal grid behind the bars makes the heights readable without competing with them
    ax.grid(axis="y", color="#eef2f7")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] wrote {save_path}")

    return save_path


def make_all_plots(csv_path, out_dir):
    """Draw one chart per measurement from the static results and return the list of files written."""
    out_dir = Path(out_dir)

    # The same collapsing and masking the summary table uses, so a chart shows one value per map and a failed run does not pull an algorithm's path length down towards zero
    df = mask_failures(collapse_repeats(load(csv_path)))

    written = []

    for metric, ylabel, title, higher_is_better in CHARTS:
        written.append(
            bar_chart(
                df,
                metric,
                ylabel,
                title,
                out_dir / f"chart_{metric}.png",
                higher_is_better=higher_is_better,
            )
        )

    return written


def plot_dynamic(csv_path, out_dir):
    """
    Draw the dynamic results, one panel per obstacle speed, with two bars per driver: how often it arrived without being hit and how often the obstacle caught it.
    """
    out_dir = Path(out_dir)
    df = pd.read_csv(csv_path)

    # A file from before the speed factor existed is drawn as a single panel
    if "speed_factor" not in df.columns:
        df = df.assign(speed_factor=1.0)

    factors = sorted(df["speed_factor"].unique())
    drivers = list(df["driver"].unique())
    positions = np.arange(len(drivers))
    width = 0.38

    fig, axes = plt.subplots(1, len(factors), figsize=(5 * len(factors), 5), sharey=True, squeeze=False)

    for ax, factor in zip(axes[0], factors):
        block = df[df["speed_factor"] == factor]
        success = [block[block["driver"] == driver]["success"].mean() for driver in drivers]
        collided = [block[block["driver"] == driver]["collided"].mean() for driver in drivers]

        ax.bar(positions - width / 2, success, width, label="reached the goal untouched", color="#2a9d8f")
        ax.bar(positions + width / 2, collided, width, label="hit by the obstacle", color="#b3402f")

        ax.set_xticks(positions)

        # The driver names are long, so they are angled to stop them overlapping
        ax.set_xticklabels(drivers, rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"obstacle at {factor:.2g}x vehicle speed", fontsize=11)
        ax.grid(axis="y", color="#eef2f7")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0][0].set_ylabel("share of runs")
    axes[0][0].legend(fontsize=9)
    fig.suptitle("Dynamic scenario: reacting against replanning", fontsize=12)

    fig.tight_layout()
    save_path = out_dir / "chart_dynamic.png"
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] wrote {save_path}")

    return save_path


def plot_goal_offset(df, save_path):
    """
    Draw how the planners respond to the goal being pulled off the diagonal: success rate on the left, median drive time when slowing for the turns on the right, one line per planner, pooled across the scenarios in the sweep.
    """
    offsets = sorted(df["value"].unique())
    algorithms = list(df["algorithm"].unique())

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))

    for algorithm in algorithms:
        block = df[df["algorithm"] == algorithm]
        colour = COLOURS.get(algorithm, "#333333")

        success = [block[block["value"] == offset]["success"].mean() for offset in offsets]
        left.plot(offsets, success, marker="o", color=colour, label=algorithm)

        # The drive is only defined on the runs that arrived
        solved = block[block["success"] == 1]
        drive = [solved[solved["value"] == offset]["drive_time_curved"].median() for offset in offsets]
        right.plot(offsets, drive, marker="o", color=colour, label=algorithm)

    left.set_ylabel("success rate")
    left.set_ylim(0, 1.05)
    right.set_ylabel("median drive time, slowing for turns (s)")

    for ax in (left, right):
        ax.set_xlabel("goal offset from the diagonal (cells)")
        ax.set_xticks(offsets)
        ax.grid(color="#eef2f7")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    left.legend(fontsize=8)
    fig.suptitle("Does the goal's place on the map decide the result?", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] wrote {save_path}")

    return save_path


def plot_sweep(long, parameter, metrics, save_path):
    """
    Draw one panel per metric from a long sweep file: the parameter's value along x, the median across maps on y, the interquartile range as a shaded band, one line per planner. The caption says how many maps went in and how many failed runs were left out of the route metrics.
    """
    values = sorted(long["value"].unique())
    planners = list(long["planner"].unique())

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.2), squeeze=False)

    excluded = 0
    for ax, metric in zip(axes[0], metrics):
        block = long[long["metric"] == metric]
        for planner in planners:
            rows = block[block["planner"] == planner]
            by_value = rows.groupby("value")["result"]

            # A rate is drawn as the rate with its Wilson interval; a measurement as the median with the quartiles. The median of a column of ones and zeros says nothing
            if metric in RATES:
                centre = by_value.mean().reindex(values)
                counts = by_value.agg(["sum", "count"]).reindex(values)
                intervals = [wilson_interval(int(row["sum"]), int(row["count"])) for _, row in counts.iterrows()]
                low = np.array([interval[0] for interval in intervals])
                high = np.array([interval[1] for interval in intervals])
            else:
                centre = by_value.median().reindex(values)
                low = by_value.quantile(0.25).reindex(values).values
                high = by_value.quantile(0.75).reindex(values).values
                excluded = max(excluded, int(by_value.apply(lambda r: r.isna().sum()).max()))

            colour = COLOURS.get(planner, "#333333")
            ax.plot(values, centre.values, marker="o", color=colour, label=planner)
            ax.fill_between(values, low, high, color=colour, alpha=0.15, linewidth=0)

        ax.set_title(metric, fontsize=10)
        ax.set_xlabel(parameter)
        ax.set_xticks(values)
        ax.grid(color="#eef2f7")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0][0].set_ylabel("rate with 95% Wilson band, or median with interquartile band")
    axes[0][0].legend(fontsize=8)

    n_maps = long.groupby(["planner", "value", "metric"]).size().max()
    fig.suptitle(
        f"{parameter}: {n_maps} maps per value; route metrics leave out the failed runs, up to {excluded} of {n_maps} at one value",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] wrote {save_path}")

    return save_path
