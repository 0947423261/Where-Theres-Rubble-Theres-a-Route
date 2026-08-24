"""
experiments/golden.py

The regression check: a frozen copy of the results table, and the comparison of a fresh run against it.

Every job ends by running this. The planners are deterministic given a seed, so a fresh run of the same configuration has to reproduce the frozen table to the last digit. A row that moved means a change altered a path, and that is either a bug or a deliberate change that has to be flagged before the table is regenerated. Timing is the one column left out, since it depends on the machine and not on the seed.

    python -m planning.experiments.golden --write --workers 8
    python -m planning.experiments.golden --check-against results/golden.csv --workers 8

The table covers all nine static planners on thirty maps per scenario and the five dynamic drivers on ten corridors at each obstacle speed, at the default configuration. The dynamic rows live next to the static ones in golden_dynamic.csv.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import Config
from .record import DYNAMIC_FIELDS, STATIC_FIELDS, save_rows
from .runner import DynamicStudy, StaticStudy, build_algorithms

# Maps per scenario in the frozen table. Fewer than the full experiment so the check finishes in minutes, enough that every planner meets every kind of map
GOLDEN_INSTANCES = 30

# Corridors in the frozen dynamic table
DYNAMIC_INSTANCES = 10

# Columns that depend on the machine rather than the seed, so they are never compared
IGNORED = ("comp_time",)

# How far a number may drift before it counts as moved. The recorded values are already rounded, so this only forgives the last bit of a float
TOLERANCE = 1e-6

# The columns that identify one row in each table
STATIC_KEYS = ["scenario", "algorithm", "instance", "repeat"]
DYNAMIC_KEYS = ["instance", "speed_factor", "driver"]


def golden_config(out_dir):
    """The configuration the frozen table was made with: every default, thirty maps per scenario, one repeat. Repeat zero is the seed every earlier row had, so the table stays comparable across the change that added repeats."""
    return Config(n_instances=GOLDEN_INSTANCES, n_repeats=1, out_dir=out_dir)


def collect(workers, out_dir):
    """Run everything the frozen table covers and return (static, dynamic) as pandas tables."""
    config = golden_config(out_dir)

    static = StaticStudy(
        config, build_algorithms(ablation=True, smoothers=True), workers=workers
    ).run()
    dynamic, _ = DynamicStudy(config, n_instances=DYNAMIC_INSTANCES).run()

    return (
        pd.DataFrame(static, columns=STATIC_FIELDS),
        pd.DataFrame(dynamic, columns=DYNAMIC_FIELDS),
    )


def shared_keys(golden, fresh, keys):
    """The key columns both tables carry. A frozen table written before a key column existed is still compared on the keys it has."""
    return [key for key in keys if key in golden.columns and key in fresh.columns]


def compare(golden, fresh, keys):
    """
    Return a list of differences between two tables, one line each, empty when the fresh table reproduces the golden one.

    Rows are matched on the key columns. A row missing from either side is a difference, and so is any compared column that moved by more than the tolerance.
    """
    differences = []

    merged = golden.merge(fresh, on=keys, how="outer", suffixes=("_golden", "_fresh"), indicator=True)

    for _, row in merged.iterrows():
        label = " ".join(f"{key}={row[key]}" for key in keys)

        if row["_merge"] == "left_only":
            differences.append(f"{label}: missing from the fresh run")
            continue
        if row["_merge"] == "right_only":
            differences.append(f"{label}: not in the golden table")
            continue

        for column in golden.columns:
            if column in keys or column in IGNORED:
                continue

            old = row[f"{column}_golden"]
            new = row[f"{column}_fresh"]

            # Numbers are compared within the tolerance, anything else exactly
            if isinstance(old, (int, float, np.integer, np.floating)):
                moved = abs(float(old) - float(new)) > TOLERANCE
            else:
                moved = old != new

            if moved:
                differences.append(f"{label} {column}: golden {old} fresh {new}")

    return differences


def write(golden_path, workers):
    """Run the full set and freeze it as the golden table."""
    golden_path = Path(golden_path)
    golden_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as scratch:
        static, dynamic = collect(workers, scratch)

    save_rows(static.to_dict("records"), golden_path, STATIC_FIELDS)
    save_rows(dynamic.to_dict("records"), dynamic_path(golden_path), DYNAMIC_FIELDS)


def check(golden_path, workers):
    """Run the full set again and report every row that moved. Returns True when nothing did."""
    golden_path = Path(golden_path)

    with tempfile.TemporaryDirectory() as scratch:
        static, dynamic = collect(workers, scratch)

    golden_static = pd.read_csv(golden_path)
    golden_dynamic = pd.read_csv(dynamic_path(golden_path))

    differences = compare(golden_static, static, shared_keys(golden_static, static, STATIC_KEYS))
    differences += compare(golden_dynamic, dynamic, shared_keys(golden_dynamic, dynamic, DYNAMIC_KEYS))

    if differences:
        print(f"\n[golden] {len(differences)} differences against {golden_path}:")
        for line in differences:
            print("  " + line)
        return False

    print(f"\n[golden] fresh run matches {golden_path} on every compared column")
    return True


def dynamic_path(golden_path):
    """The dynamic table sits next to the static one with _dynamic in the name."""
    golden_path = Path(golden_path)
    return golden_path.with_name(golden_path.stem + "_dynamic" + golden_path.suffix)


def main():
    parser = argparse.ArgumentParser(description="Freeze the results table, or check a fresh run against the frozen one.")
    parser.add_argument("--write", action="store_true", help="run the full set and write results/golden.csv")
    parser.add_argument("--check-against", metavar="CSV", help="run the full set and compare it with this golden table")
    parser.add_argument("--workers", type=int, default=1, help="processes to spread the maps across. Timing is not compared, so more is fine")
    parser.add_argument("--out", default="results/golden.csv", help="where --write puts the table")
    arguments = parser.parse_args()

    if not arguments.write and not arguments.check_against:
        parser.error("choose --write or --check-against.")

    if arguments.write:
        write(arguments.out, arguments.workers)

    if arguments.check_against:
        if not check(arguments.check_against, arguments.workers):
            sys.exit(1)


if __name__ == "__main__":
    main()
