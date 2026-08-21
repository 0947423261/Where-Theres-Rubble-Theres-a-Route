"""
experiments/runner.py

Runs the experiments and collects the rows that the analysis and the charts are built from.

There are two studies. The static study drives every algorithm across every instance of every static scenario, which is the paired comparison the results chapter is built on. The dynamic study drives a smaller set of vehicles through the moving obstacle scenario, where what matters is whether the vehicle got through rather than how tidy its route was.

Every algorithm sees the same maps, and the seed for a stochastic run is derived from the scenario, the instance and the algorithm name. That means a single number in the configuration reproduces the whole experiment, and rerunning one algorithm on one map gives back exactly the run that was recorded.
"""

import hashlib
import os
import time
from multiprocessing import Pool

import numpy as np

from ..algorithms.apf import APF
from ..algorithms.dynamic import PlanOnceDriver, ReactiveAPF
from ..algorithms.hybrid import Hybrid
from ..algorithms.rrt_star import RRTStar
from ..algorithms.smoothers import (
    BezierSmoother,
    BSplineSmoother,
    DubinsSmoother,
    NoSmoother,
)
from ..maps.moving_obstacle import make_dynamic_instance
from ..maps.scenarios import make_scenario
from ..utils.collision import checks_done, reset_checks
from ..utils.metrics import path_length, score_run, time_to_detection
from ..utils.optimal import astar_length
from ..utils.visualise import draw_path
from .record import DYNAMIC_FIELDS, STATIC_FIELDS, dynamic_row, save_rows, static_row


def _quieten_worker():
    """
    Stop each worker from starting threads of its own.

    Numpy and scipy will happily open a thread per core inside a single process. With a worker already running on every core that oversubscribes the machine and makes everything slower, so each worker is told to keep to itself.
    """
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"


def run_seed(master_seed, scenario, instance, name, repeat):
    """
    The seed for repeat k of one planner on one map.

    Repeat zero is seeded from the four parts every run used before repeats existed, so a row recorded then is reproduced by k = 0 and the golden table still lines up. Later repeats add k as a fifth part.
    """
    if repeat == 0:
        return derived_seed(master_seed, scenario, instance, name)

    return derived_seed(master_seed, scenario, instance, name, repeat)


def junction_points(result):
    """The places where a hybrid handed control to RRT*, as points. The indices in the result refer to the route before smoothing, so they are looked up there."""
    route = result.get("raw_path", result.get("path"))
    if route is None:
        return []
    return [route[j] for j in result.get("junctions", []) if 0 <= j < len(route)]


def derived_seed(*parts):
    """
    Turn any set of identifiers into a seed.

    The parts are joined into one string and hashed. Python's own hash is randomised per process, so a run seeded with it could never be reproduced on a later day, which is exactly what a reproducible experiment cannot afford.
    """
    # Join the parts with a separator that cannot appear inside them, so ("ab", "c") and ("a", "bc") do not collide
    key = "|".join(str(part) for part in parts).encode()

    # Take the first eight hex digits of the digest, which is a 32 bit number and the largest a numpy generator accepts as a plain seed
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


class PlannerSpec:
    """
    One entry in the study. It pairs the name that appears in the results with the planner class and the options that give that entry its behaviour, so the runner never has to know which algorithm it is running.
    """

    def __init__(self, name, planner_class, adaptive=None, smoother_class=None, post_smoother_class=None, seed_name=None, deterministic=False):
        # The name used in the CSV, the charts and the figure filenames
        self.name = name

        # The class to construct for each run
        self.planner_class = planner_class

        # Whether the hybrid uses adaptive stuck detection. It stays None for planners that have no such option
        self.adaptive = adaptive

        # Which smoothing strategy the hybrid uses. It stays None for planners that do not smooth
        self.smoother_class = smoother_class

        # A smoother applied to the finished route by this runner rather than inside the planner. This is how RRT* gets a smoothed row without rrt_star.py knowing about smoothing
        self.post_smoother_class = post_smoother_class

        # The name the run is seeded by. Usually the entry's own name, but an entry that post-processes another planner's route is seeded as that planner, so the two rows differ only by the post-processing
        self.seed_name = seed_name if seed_name is not None else name

        # A deterministic planner gives the same route on every seed, so it is run once per map however many repeats the configuration asks for
        self.deterministic = deterministic

    def build(self, grid, start, goal, config, rng, trace=False):
        """Construct the planner object for one run. A fresh object is built every time because the planners hold per-run state. trace asks the planners that can for their tree history, which only the animation wants."""
        # Only pass the options this planner actually accepts
        kwargs = {}

        if trace and self.planner_class in (RRTStar, Hybrid):
            kwargs["trace"] = True

        if self.adaptive is not None:
            kwargs["adaptive"] = self.adaptive

        if self.smoother_class is not None:
            # The smoother reads its own settings out of the configuration, so it is constructed here rather than shared between runs
            kwargs["smoother"] = self.smoother_class(config)

        return self.planner_class(grid, start, goal, config, rng, **kwargs)

    def plan(self, grid, start, goal, config, rng, trace=False):
        """Build the planner, run it, and apply the post smoother to a successful route. Returns the planner's result dictionary."""
        result = self.build(grid, start, goal, config, rng, trace=trace).plan()

        # A route that was never found has nothing to smooth. The route before smoothing is kept so the animation can overlay the two
        if self.post_smoother_class is not None and result["success"] and result["path"]:
            result["raw_path"] = result["path"]
            result["path"] = self.post_smoother_class(config).smooth(result["path"], grid)

        return result


def build_algorithms(ablation=False, smoothers=False):
    """
    Return the list of entries to run.

    The five core entries are the two standalone planners, RRT* with its route smoothed afterwards, the standard hybrid and the modified hybrid. The ablation adds the two variants that isolate which of the two extensions did the work, and the smoother comparison adds the other two smoothing strategies on top of the modified hybrid.
    """
    algorithms = [
        # The two baselines, each on its own
        PlannerSpec("APF", APF, deterministic=True),
        PlannerSpec("RRT*", RRTStar),
        # The same RRT* route with its corners rounded afterwards, seeded as RRT* so it is the same tree. This is the fair comparison for the modified hybrid's smoothing: a smoothed global planner against a smoothed hybrid
        PlannerSpec("RRT*+Bezier", RRTStar, post_smoother_class=BezierSmoother, seed_name="RRT*"),
        # The standard hybrid: fixed stuck window and no smoothing
        PlannerSpec("Hybrid", Hybrid, adaptive=False, smoother_class=NoSmoother),
        # The full method: adaptive stuck detection and Bezier smoothing
        PlannerSpec("Modified", Hybrid, adaptive=True, smoother_class=BezierSmoother),
    ]

    if ablation:
        algorithms += [
            # Adaptive stuck detection on its own, which shows what the detection contributes
            PlannerSpec("Adaptive", Hybrid, adaptive=True, smoother_class=NoSmoother),
            # Smoothing on its own, which shows what the smoothing contributes
            PlannerSpec("Bezier", Hybrid, adaptive=False, smoother_class=BezierSmoother),
        ]

    if smoothers:
        algorithms += [
            # The same modified hybrid with the other two smoothing strategies, which compares the strategies against each other rather than against no smoothing
            PlannerSpec("BSpline", Hybrid, adaptive=True, smoother_class=BSplineSmoother),
            PlannerSpec("Dubins", Hybrid, adaptive=True, smoother_class=DubinsSmoother),
        ]

    return algorithms


def _run_one_instance(task):
    """
    Run every algorithm on one map and return its rows.

    This sits at module level rather than inside the class because a worker process receives its task by unpickling it, and a bound method of a live object is not something that survives that trip. The configuration and the algorithm list are plain data and classes, so they do.
    """
    config, algorithms, scenario, instance, save_figures = task

    return StaticStudy(config, algorithms, save_figures).run_instance(scenario, instance)


class StaticStudy:
    """Drives every algorithm across every instance of every static scenario and collects one row per run."""

    def __init__(self, config, algorithms, save_figures=False, workers=1):
        self.config = config
        self.algorithms = algorithms

        # Whether to save one picture per scenario and algorithm from the first instance
        self.save_figures = save_figures

        # How many processes to spread the maps across. Every run is seeded from the master seed rather than drawn from a shared stream, so the rows come back identical however many workers there are.
        #
        # The one thing that does not survive is the timing. Workers on separate cores compete for memory bandwidth, so comp_time comes out inflated and uneven, and mission_time is built on comp_time. Use more than one worker while you are iterating, and take the numbers that go in the results from a single worker run
        self.workers = max(1, int(workers))

    def run_one(self, spec, grid, start, goal, seed):
        """Run one planner on one map and return (result, elapsed seconds, collision checks)."""
        # A fresh generator per run, seeded so the run can be reproduced on its own
        rng = np.random.default_rng(seed)

        # The check counter starts from zero so the number read back is this run's alone. It is read before the scorer runs, so the scorer's own checks are not billed to the planner
        reset_checks()

        # Construction is timed along with planning, since building the distance fields is part of the cost of using the algorithm
        started = time.perf_counter()
        result = spec.plan(grid, start, goal, self.config, rng)
        elapsed = time.perf_counter() - started

        return result, elapsed, checks_done()

    def run_instance(self, scenario, instance):
        """Run every algorithm on one map and return the rows for it. This is the unit of work a worker process is handed."""
        rows = []

        # Every algorithm sees this same map, which is what makes the comparison paired. The generator comes back too, since the blocked road one knows where its pocket is
        grid, start, goal, generator = make_scenario(
            scenario, self.config.master_seed + instance, self.config
        )
        pocket = getattr(generator, "pocket", None)

        # The shortest route on this map, found once since it is deterministic. Every algorithm's length is reported as a ratio to it
        optimum = astar_length(grid, start, goal, self.config.goal_tolerance)

        for spec in self.algorithms:
            # A stochastic planner is run on several seeds per map so the analysis can take the map's median and report the spread. A deterministic one would give the same row every time, so it runs once
            repeats = 1 if spec.deterministic else max(1, int(self.config.n_repeats))

            for repeat in range(repeats):
                rows.append(self._run_repeat(spec, grid, start, goal, scenario, instance, repeat, optimum, pocket))

            # One example figure per scenario and algorithm, taken from the first instance and the first repeat
            if self.save_figures and instance == 0:
                seed = run_seed(self.config.master_seed, scenario, instance, spec.seed_name, 0)
                result, _, _ = self.run_one(spec, grid, start, goal, seed)
                draw_path(
                    grid,
                    start,
                    goal,
                    result["path"],
                    title=f"{spec.name} on {scenario}",
                    save_path=self.config.out_dir
                    / f"example_{scenario}_{spec.name.replace('*', 'star')}.png",
                    junction_points=junction_points(result),
                )

        return rows

    def _run_repeat(self, spec, grid, start, goal, scenario, instance, repeat, optimum, pocket):
        """Run one planner once on one map and return its row."""
        # The seed for this run, unique per scenario, instance, algorithm and repeat but derived from the master seed. An entry that post-processes another planner is seeded as that planner
        seed = run_seed(self.config.master_seed, scenario, instance, spec.seed_name, repeat)

        result, elapsed, checks = self.run_one(spec, grid, start, goal, seed)

        # Score the run against the same checks for every algorithm
        scores = score_run(
            grid, result["path"], goal, self.config, result["success"]
        )

        return static_row(
                    scenario=scenario,
                    algorithm=spec.name,
                    instance=instance,
                    repeat=repeat,
                    comp_time=elapsed,
                    switches=result.get("switches", 0),
                    collision_checks=checks,
                    astar_length=optimum,
                    detection=time_to_detection(
                        result.get("raw_path", result["path"]), result.get("junctions", []), pocket
                    ),
                    scores=scores,
                )

    def run(self):
        """Run the whole static study and return the list of rows, spread across processes when more than one worker was asked for."""
        if self.workers > 1:
            return self._run_across_workers()

        rows = []

        for scenario in self.config.scenarios:
            print(f"=== scenario: {scenario} ===")

            for instance in range(self.config.n_instances):
                rows += self.run_instance(scenario, instance)
                print(f"  instance {instance + 1}/{self.config.n_instances} done")

            print()

        return rows

    def _run_across_workers(self):
        """
        Run the maps in a pool of processes. Each task is one map with every algorithm on it, which keeps the tasks large enough that handing them to a worker costs far less than the work itself.

        The rows come back sorted, so the results file is byte for byte what a single worker would have written apart from the timing columns.
        """
        print(f"[runner] running across {self.workers} workers. Treat the timing columns as unusable")

        # One task per map. The configuration and the algorithm list travel with it, since a worker starts with nothing
        tasks = [
            (self.config, self.algorithms, scenario, instance, self.save_figures)
            for scenario in self.config.scenarios
            for instance in range(self.config.n_instances)
        ]

        rows = []

        with Pool(processes=self.workers, initializer=_quieten_worker) as pool:
            for done, instance_rows in enumerate(
                pool.imap_unordered(_run_one_instance, tasks), start=1
            ):
                rows += instance_rows
                print(f"  {done}/{len(tasks)} maps done")

        # The pool returns whichever map finished first, so put the rows back in a fixed order. The sort is stable, so the algorithms and repeats inside a map keep the order a single worker wrote them in
        order = {name: i for i, name in enumerate(self.config.scenarios)}
        rows.sort(key=lambda row: (order[row["scenario"]], row["instance"]))

        return rows

    def save(self, rows):
        """Write the static rows out and return the path of the file."""
        return save_rows(rows, self.config.out_dir / "raw_results.csv", STATIC_FIELDS)


class DynamicStudy:
    """
    Drives a smaller set of vehicles through the moving obstacle scenario.

    The vehicles are the reactive APF driver and the plan-once drivers, each of the latter with and without replanning. A replanning driver plans again before contact when its route ahead is threatened, and after a hit. All of them meet the same obstacle following the same walk on a given instance, because the obstacle remembers where it was and is shared between the drivers on that map.

    The obstacle's speed is a factor of the study: every walk is met at each of the configured speeds relative to the vehicle, and every driver meets all of them. The driver's own seed does not change with the speed, so the first route it plans is the same at every speed and the speed is the only thing that differs between those rows.
    """

    def __init__(self, config, n_instances=None, first_instance=0):
        self.config = config

        # How many walks, from the configuration unless the caller wants fewer
        self.n_instances = n_instances if n_instances is not None else config.dynamic_instances

        # Which instance to start counting from, so a single encounter can be replayed without running the ones before it
        self.first_instance = first_instance

    def build_drivers(self):
        """
        Return the list of (name, factory) pairs to drive with. Each factory takes the map, the two ends and the obstacle and returns the driver object.
        """
        return [
            # APF recomputed every step, which never commits to a route
            (
                "APF reactive",
                lambda grid, start, goal, rng, obstacle: ReactiveAPF(
                    grid, start, goal, self.config, rng, obstacle
                ),
            ),
            # RRT* planned once over the static map and then driven blind
            (
                "RRT* plan once",
                lambda grid, start, goal, rng, obstacle: PlanOnceDriver(
                    grid, start, goal, self.config, rng, obstacle, RRTStar, replan=False
                ),
            ),
            # The same planner, planning again from wherever the vehicle stands whenever the route ahead runs within reach of the obstacle, and after any contact
            (
                "RRT* replan",
                lambda grid, start, goal, rng, obstacle: PlanOnceDriver(
                    grid, start, goal, self.config, rng, obstacle, RRTStar, replan=True
                ),
            ),
            # The modified hybrid planned once, which shows that the extensions do nothing about a world that moves
            (
                "Modified plan once",
                lambda grid, start, goal, rng, obstacle: PlanOnceDriver(
                    grid, start, goal, self.config, rng, obstacle, Hybrid,
                    replan=False, adaptive=True, smoother=BezierSmoother(self.config),
                ),
            ),
            # The modified hybrid with replanning, which recovers but pays for every recovery
            (
                "Modified replan",
                lambda grid, start, goal, rng, obstacle: PlanOnceDriver(
                    grid, start, goal, self.config, rng, obstacle, Hybrid,
                    replan=True, adaptive=True, smoother=BezierSmoother(self.config),
                ),
            ),
        ]

    def run(self):
        """Run the whole dynamic study and return (rows, examples), where examples holds one run per driver for the animations."""
        rows = []

        # One saved run per driver from the first instance, kept so the animation script does not have to run the study again
        examples = {}

        drivers = self.build_drivers()

        # The speed the animations are taken at: the vehicle's own, or the first level if that is not among them
        factors = tuple(self.config.dynamic_speed_factors)
        example_factor = 1.0 if 1.0 in factors else factors[0]

        for instance in range(
            self.first_instance, self.first_instance + self.n_instances
        ):
            for speed_factor in factors:
                # The map and the obstacle for this instance and speed. The obstacle is shared across the drivers so every one of them meets the same walk
                grid, start, goal, obstacle = make_dynamic_instance(
                    self.config.master_seed + instance, self.config, speed_factor
                )

                for name, factory in drivers:
                    rng = np.random.default_rng(
                        derived_seed(self.config.master_seed, "dynamic", instance, name)
                    )

                    # The driver times its own planning calls, since the driving loop around them is not planning. The check counter covers the whole run: every cell asked about while planning or driving is the driver's cost
                    reset_checks()
                    result = factory(grid, start, goal, rng, obstacle).plan()

                    rows.append(
                        dynamic_row(
                            instance=instance,
                            speed_factor=speed_factor,
                            driver=name,
                            result=result,
                            comp_time=result["plan_time"],
                            collision_checks=checks_done(),
                            path_length=path_length(result["path"]),
                        )
                    )

                    # Keep the first instance of each driver, at the example speed, for the animations
                    if instance == self.first_instance and speed_factor == example_factor:
                        examples[name] = (grid, start, goal, obstacle, result)

                    outcome = (
                        "reached the goal"
                        if result["success"]
                        else ("was hit" if result["collided"] else "did not arrive")
                    )
                    print(
                        f"  instance {instance} at {speed_factor:.2g}x: {name} {outcome} in {result['iters']} steps"
                        f" with {result['replans']} replans"
                    )

        return rows, examples

    def save(self, rows):
        """Write the dynamic rows out and return the path of the file."""
        return save_rows(rows, self.config.out_dir / "dynamic_results.csv", DYNAMIC_FIELDS)
