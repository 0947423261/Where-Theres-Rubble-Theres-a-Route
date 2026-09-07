"""
main.py

The entry point that runs the whole comparison

This is the file to run. It builds the maps, drives every algorithm across all of them, saves the raw results, then produces the summary table, the significance tests and the charts.

Run it from the folder that contains this file:

    python main.py --quick                  a small fast run for checking the code works
    python main.py --full                   the full experiment using the values in the configuration
    python main.py --full --ablation        also run the two variants that isolate each extension
    python main.py --full --smoothers       also run the B-spline and Dubins smoothing strategies
    python main.py --full --figures         also save one example picture per scenario and algorithm
    python main.py --quick --dynamic        also run the moving obstacle study

The quick run shrinks the grid, the number of maps and the iteration budgets so the whole loop finishes in seconds. Nothing from a quick run should appear in the results chapter, since the smaller budgets change how often each algorithm succeeds.
"""

import argparse
from pathlib import Path

from planning.experiments.analyse import run_full_analysis, summarise_dynamic
from planning.experiments.plot import make_all_plots, plot_dynamic
from planning.experiments.runner import DynamicStudy, StaticStudy, build_algorithms
from planning.utils.config import Config


def build_config(quick):
    """Return the configuration for this run, shrunk down when the quick flag was given."""
    # The full experiment uses every default in utils/config.py
    if not quick:
        return Config()

    # The quick run is for checking that the code still works end to end, so everything is small
    return Config(
        grid_size=40,
        n_instances=3,
        n_repeats=1,
        apf_max_iter=1500,
        rrt_max_iter=1200,
        start=(2, 3),
        goal=(37, 36),
    )


def parse_arguments():
    """Read the command line switches."""
    parser = argparse.ArgumentParser(
        description="Compare APF, RRT* and the hybrid planners on post-disaster street maps."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small fast run on a smaller grid with fewer maps, for checking the code",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="the full experiment using the values in the configuration",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="also run the adaptive-only and smoothing-only variants",
    )
    parser.add_argument(
        "--smoothers",
        action="store_true",
        help="also run the B-spline and Dubins smoothing strategies",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="also save one example picture per scenario and algorithm",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="how many processes to spread the maps across. More than one makes the timing columns unusable",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="also run the moving obstacle study",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=None,
        help="override how many maps each static scenario uses",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="override how many seeds each stochastic planner runs per map. The analysis pairs on the per-map median",
    )
    parser.add_argument(
        "--dynamic-instances",
        type=int,
        default=None,
        help="how many obstacle walks the moving obstacle study uses, each met at every configured speed",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="the master seed. The headline run uses one the design was never fitted on, and says which",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="where the results, tables and charts go; outputs/ otherwise",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    # The quick flag wins when both are given, since it is the safer of the two to run by accident
    quick = arguments.quick or not arguments.full
    config = build_config(quick)

    # An explicit count wins over whichever default the quick or the full mode chose
    if arguments.instances is not None:
        config.n_instances = arguments.instances
    if arguments.repeats is not None:
        config.n_repeats = arguments.repeats
    if arguments.seed is not None:
        config.master_seed = arguments.seed
    if arguments.out is not None:
        config.out_dir = Path(arguments.out)
        config.out_dir.mkdir(parents=True, exist_ok=True)

    algorithms = build_algorithms(
        ablation=arguments.ablation, smoothers=arguments.smoothers
    )

    print("[main] quick run" if quick else "[main] full run")
    print(f"[main] algorithms: {[spec.name for spec in algorithms]}")
    print(f"[main] scenarios:  {list(config.scenarios)}")
    print(f"[main] seed:       {config.master_seed}")
    print(f"[main] instances:  {config.n_instances} per scenario")
    print(f"[main] repeats:    {config.n_repeats} per stochastic planner and map\n")

    ### === The static comparison === ###
    study = StaticStudy(
        config, algorithms, save_figures=arguments.figures, workers=arguments.workers
    )
    csv_path = study.save(study.run())

    run_full_analysis(csv_path, config.out_dir)
    make_all_plots(csv_path, config.out_dir)
    ### === END ===

    ### === The moving obstacle study === ###
    if arguments.dynamic:
        dynamic = DynamicStudy(config, n_instances=arguments.dynamic_instances)
        rows, _ = dynamic.run()
        dynamic_csv = dynamic.save(rows)

        summarise_dynamic(dynamic_csv, config.out_dir)
        plot_dynamic(dynamic_csv, config.out_dir)
    ### === END ===

    print(f"\n[main] finished. The files are in {config.out_dir}:")
    print("  raw_results.csv         every static run")
    print("  summary_table.csv       means and standard deviations")
    print("  significance_tests.csv  the modified hybrid against the others")
    print("  chart_*.png             one chart per measurement")
    if arguments.figures:
        print("  example_*.png           example routes")
    if arguments.dynamic:
        print("  dynamic_results.csv     every run against the moving obstacle")
        print("  dynamic_summary.csv     means per way of driving")


# Only run when this file is executed directly, so importing from it stays free of side effects
if __name__ == "__main__":
    main()
