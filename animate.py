"""
animate.py

Makes one animation of one planner driving one map

Builds the map, runs the algorithm you name and writes an animation of the vehicle driving the route it found.

Run it from the folder that contains this file:

    python animate.py --scenario blocked_road --algorithm APF
    python animate.py --scenario blocked_road --algorithm Modified
    python animate.py --scenario dense_city --algorithm RRT* --instance 7
    python animate.py --scenario dense_city --algorithm Modified --format mp4

The first two lines are the pair worth showing side by side: APF drives into the collapse and stops, and the modified hybrid escapes the same pocket and carries on. Both use the same map because the instance number and the master seed decide the map, not the algorithm.

The seeding matches main.py exactly, so the run being animated is the same run that was recorded in the results.

MP4 needs ffmpeg installed on the machine. GIF works everywhere and plays inside a slide deck.
"""

import argparse

import numpy as np

from planning.algorithms.apf import APF
from planning.algorithms.hybrid import Hybrid
from planning.experiments.runner import build_algorithms, derived_seed
from planning.maps.scenarios import SCENARIOS, make_instance
from planning.utils.animate import animate_run
from planning.utils.config import Config


def parse_arguments(names):
    """Read the command line switches. The algorithm names come from the same place the experiment builds them, so the two can never drift apart."""
    parser = argparse.ArgumentParser(description="Animate one planner run.")
    parser.add_argument("--scenario", default="blocked_road", choices=list(SCENARIOS))
    parser.add_argument("--algorithm", default="Modified", choices=names)
    parser.add_argument(
        "--instance", type=int, default=0, help="which of the seeded maps to use"
    )
    parser.add_argument("--format", default="gif", choices=["gif", "mp4"])
    parser.add_argument(
        "--frames", type=int, default=150, help="more frames give smoother playback and a larger file"
    )
    parser.add_argument("--fps", type=int, default=25, help="playback speed")
    return parser.parse_args()


def main():
    config = Config()

    # Every algorithm the study knows about, including the ablation variants and the other smoothers
    algorithms = {
        spec.name: spec for spec in build_algorithms(ablation=True, smoothers=True)
    }

    arguments = parse_arguments(list(algorithms))
    spec = algorithms[arguments.algorithm]

    # The same map main.py would build for this instance number
    grid, start, goal = make_instance(
        arguments.scenario, config.master_seed + arguments.instance, config
    )

    # The same seed main.py would give this run, so the animation shows the recorded run rather than a fresh one
    rng = np.random.default_rng(
        derived_seed(
            config.master_seed, arguments.scenario, arguments.instance, spec.seed_name
        )
    )

    # The same run as the results hold, with the tree history recorded for the picture. Recording does not touch the random draws, so it is the same run
    result = spec.plan(grid, start, goal, config, rng, trace=True)

    outcome = "reached the goal" if result["success"] else "did not reach the goal"
    print(
        f"[animate] {spec.name} on {arguments.scenario} instance {arguments.instance}: {outcome}"
    )

    # A planner that failed early can leave too little of a route to animate, which is worth saying rather than crashing on
    if result["path"] is None or len(result["path"]) < 2:
        print("[animate] there is no route to animate. Try another algorithm or another instance.")
        return

    # A filename that says what it shows, with the star in RRT* replaced since it is a wildcard in most shells
    stem = f"{arguments.scenario}_{spec.name}_i{arguments.instance}".replace("*", "star")

    animate_run(
        grid,
        start,
        goal,
        result,
        config.out_dir / f"anim_{stem}.{arguments.format}",
        title=f"{spec.name} on {arguments.scenario}",
        n_frames=arguments.frames,
        fps=arguments.fps,
        config=config,
        # The field is drawn for the planners that follow it
        field=spec.planner_class in (APF, Hybrid),
    )


if __name__ == "__main__":
    main()
