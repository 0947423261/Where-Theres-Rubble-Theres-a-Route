"""
dynamic.py

Runs the moving obstacle scenario and animates it

The static maps never show what APF is for. APF recomputes its move from what it can see at that instant, so an obstacle that moves is just a field that changed, and the vehicle gives way without being told to. RRT* and the hybrid plan a route over one snapshot of the world, and when the obstacle wanders onto that route the vehicle drives straight into it.

Run it from the folder that contains this file:

    python dynamic.py                       one map, every driver, one animation each
    python dynamic.py --instance 3          a different map
    python dynamic.py --study               the whole study across several maps, with the summary and the chart
    python dynamic.py --no-animation        skip the animations and only print the outcomes

Each animation shows the vehicle and the obstacle on the same clock, so what is on screen is the encounter as it happened. A run that ended in a collision is marked with a cross where the vehicle was hit.
"""

import argparse

from planning.experiments.analyse import summarise_dynamic
from planning.experiments.plot import plot_dynamic
from planning.experiments.runner import DynamicStudy
from planning.utils.animate import animate_dynamic_run
from planning.utils.config import Config


def parse_arguments():
    """Read the command line switches."""
    parser = argparse.ArgumentParser(
        description="Drive the moving obstacle scenario and animate the result."
    )
    parser.add_argument(
        "--instance", type=int, default=0, help="which of the seeded dynamic maps to use"
    )
    parser.add_argument(
        "--study",
        action="store_true",
        help="run the whole study across several maps instead of one",
    )
    parser.add_argument(
        "--instances", type=int, default=None, help="how many obstacle walks the study uses, each met at every configured speed"
    )
    parser.add_argument("--format", default="gif", choices=["gif", "mp4"])
    parser.add_argument(
        "--no-animation", action="store_true", help="print the outcomes without writing animations"
    )
    return parser.parse_args()


def animate_examples(config, examples, file_format):
    """Write one animation per driver from the saved example runs."""
    for name, (grid, start, goal, obstacle, result) in examples.items():
        # A driver that never moved has nothing to animate
        if result["path"] is None or len(result["path"]) < 2:
            print(f"[dynamic] {name} produced no route to animate.")
            continue

        # Mark the collision on the frame it happened, which is the last step the vehicle drove
        collided_at = len(result["path"]) - 1 if result["collided"] else None

        # A filename that says which driver it shows, with the star in RRT* replaced since it is a wildcard in most shells
        stem = name.replace("*", "star").replace(" ", "_")

        animate_dynamic_run(
            grid,
            start,
            goal,
            obstacle,
            result["path"],
            config.out_dir / f"anim_dynamic_{stem}.{file_format}",
            title=f"{name} against the moving obstacle",
            collided_at=collided_at,
        )


def main():
    arguments = parse_arguments()
    config = Config()

    # The study runs several maps and produces the table and the chart. A single run is for looking at one encounter closely
    if arguments.study:
        study = DynamicStudy(config, n_instances=arguments.instances)
        rows, examples = study.run()
        csv_path = study.save(rows)

        summarise_dynamic(csv_path, config.out_dir)
        plot_dynamic(csv_path, config.out_dir)
    else:
        # Running the study over a single instance keeps one code path for both cases
        study = DynamicStudy(config, n_instances=1, first_instance=arguments.instance)
        rows, examples = study.run()

    if not arguments.no_animation:
        animate_examples(config, examples, arguments.format)


if __name__ == "__main__":
    main()
