"""
experiments/record.py

Turns the outcome of a single run into one row of numbers and writes the collected rows out as a CSV.

The raw per-run data is saved rather than only the averages. Keeping it means any statistic can be recomputed later without running the experiment again, and the paired significance tests in analyse.py need the individual runs to line up map by map.

There are two row shapes because the static runs and the dynamic runs answer different questions. A static run is scored on how good its route is, while a dynamic run is scored on whether the vehicle got through without being hit.
"""

import csv
from pathlib import Path

# The columns of the static results file, in the order they are written
STATIC_FIELDS = [
    "scenario",  # which map type the run used
    "algorithm",  # which planner ran
    "instance",  # which of the seeded maps
    "repeat",  # which seed of the planner on that map. 0 is the seed every run had before repeats existed; APF only ever has 0
    "success",  # 1 when the vehicle arrived along a collision free path, otherwise 0
    "path_length",  # total distance driven in cells
    "astar_length",  # the shortest eight-connected route on the same map, the yardstick the ratio divides by
    "length_ratio",  # path_length over astar_length, which is what the tables and charts report. 1.0 is the grid optimum
    "comp_time",  # seconds the planner spent thinking
    "smoothness",  # 0 for a straight line up to 1 for a path that doubles back constantly, measured per vertex
    "curvature",  # radians turned per cell driven, which does not depend on how finely the planner sampled the route
    "max_curvature",  # the tightest turn on the route in radians per cell, so 1 over this is the smallest turning radius the route demands
    "clearance_min",  # closest the route came to any obstacle in cells
    "clearance_mean",  # average distance from obstacles in cells
    "total_turning",  # every heading change on the route added up, in radians
    "sharp_turns",  # how many vertices turn by more than turn_threshold, which is the corners smoothing left behind
    "radius_violated",  # 1 when the route somewhere turns tighter than dubins_radius allows
    "drive_time_curved",  # seconds to drive the route when the vehicle slows for its turns, the second half of mission_time_curved
    "switches",  # number of RRT* escapes, which is 0 for APF and for RRT* on its own
    "false_alarm",  # 1 when the planner escaped at least once on a normal city map, where nothing should trap it. Empty on the other scenarios
    "time_to_detection",  # on the blocked road, path points between entering the pocket and the first escape. Empty elsewhere, and empty when the first escape came before the pocket or never came
    "collision_checks",  # how many cells the planner asked the collision helpers about, its share of the budget stated in the README
]

# The columns of the dynamic results file
DYNAMIC_FIELDS = [
    "instance",  # which of the seeded dynamic maps
    "speed_factor",  # the obstacle's speed as a multiple of the vehicle's on this run
    "driver",  # which way of driving was used
    "success",  # 1 when the vehicle arrived and was never hit
    "reached",  # 1 when it arrived at all
    "collided",  # 1 when the moving obstacle caught it
    "steps",  # time steps taken
    "replans",  # number of times the route was planned again
    "comp_time",  # seconds spent in planning calls across the whole run, the driving loop and the hit tests excluded
    "collision_checks",  # how many cells the driver asked the collision helpers about across the whole run, the moving obstacle's own overlap test excluded
    "path_length",  # total distance driven in cells
]


def static_row(scenario, algorithm, instance, repeat, comp_time, switches, collision_checks, astar_length, detection, scores):
    """
    Build one row of the static results from the identifiers of the run, its timing, its check count, the map's optimum, its time to detection and the dictionary of measurements returned by metrics.score_run.
    """
    # An escape on the normal city is the stuck detector firing where nothing should trap it. The column is left empty on the other scenarios so it averages into a rate over the normal city alone
    false_alarm = int(switches > 0) if scenario == "normal_city" else float("nan")

    # A map with no optimum has no ratio. It cannot happen for a generated map, which is checked for a route before it is handed out
    if astar_length is None or astar_length <= 0:
        ratio = float("nan")
    else:
        ratio = scores["path_length"] / astar_length

    return {
        "scenario": scenario,
        "algorithm": algorithm,
        "instance": instance,
        "repeat": int(repeat),
        # True and False are stored as 1 and 0 so the column averages straight into a success rate
        "success": int(bool(scores["success"])),
        "path_length": round(scores["path_length"], 3),
        "astar_length": round(astar_length, 3) if astar_length is not None else float("nan"),
        "length_ratio": round(ratio, 4),
        "comp_time": round(comp_time, 5),
        "smoothness": round(scores["smoothness"], 4),
        "curvature": round(scores["curvature"], 4),
        "max_curvature": round(scores["max_curvature"], 4),
        "clearance_min": round(scores["clearance_min"], 3),
        "clearance_mean": round(scores["clearance_mean"], 3),
        "total_turning": round(scores["total_turning"], 4),
        "sharp_turns": int(scores["sharp_turns"]),
        "radius_violated": int(bool(scores["radius_violated"])),
        "drive_time_curved": round(scores["drive_time_curved"], 4),
        "switches": int(switches),
        "false_alarm": false_alarm,
        "time_to_detection": detection,
        "collision_checks": int(collision_checks),
    }


def dynamic_row(instance, speed_factor, driver, result, comp_time, collision_checks, path_length):
    """
    Build one row of the dynamic results from the identifiers of the run and the result dictionary returned by one of the dynamic drivers.
    """
    return {
        "instance": instance,
        "speed_factor": float(speed_factor),
        "driver": driver,
        "success": int(bool(result["success"])),
        "reached": int(bool(result["reached"])),
        "collided": int(bool(result["collided"])),
        "steps": int(result["iters"]),
        "replans": int(result["replans"]),
        "comp_time": round(comp_time, 5),
        "collision_checks": int(collision_checks),
        "path_length": round(path_length, 3),
    }


def save_rows(rows, out_path, fieldnames):
    """Write a list of row dictionaries to a CSV file and return the path it was written to."""
    out_path = Path(out_path)

    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)

        # The first line names the columns, then one line follows per run
        writer.writeheader()
        writer.writerows(rows)

    print(f"[record] wrote {len(rows)} rows to {out_path}")

    return out_path
