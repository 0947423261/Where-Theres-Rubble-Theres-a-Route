"""
utils/config.py

File containing all tunable/configurable elements within the project.
"""

# Using dataclass to avoid boilerplate code when creating the class constructor
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ======================
    # Reproducibility
    master_seed: int = 42  # fixes the random number generator so the same maps and the same stochastic runs come back on every execution. Change it for a different but equally repeatable set of maps

    # ======================
    # Environment parameters
    grid_size: int = 90  # the map is a square of this many cells on each side. Every cell is free (0) or an obstacle (non-zero). Ninety rather than sixty because the vehicle is two cells wide: streets had to grow so a corner can be taken at the Dubins radius, and the map grew with them so a route still crosses several blocks
    start: tuple = (2, 3)  # start position as (x, y) which is (column, row). It sits on the edge street that the generator always carves
    goal: tuple = (87, 86)  # goal position as (x, y). Deliberately off the exact corner-to-corner diagonal so the attractive force does not cancel perfectly against the first building corner, which would be a pathological equilibrium. The goal offset sweep of 2026-09-15 (results/goal_offset_seed42.csv) found the results do not depend on the offset with the start also a cell off the diagonal, so this is a precaution rather than a lever
    goal_tolerance: float = (
        1.5  # the vehicle counts as arrived once it is within this many cells of the goal
    )

    # ======================
    # Experiment parameters
    n_instances: int = 100  # number of randomly generated maps per scenario. Every algorithm runs on the same maps so the comparison is paired. A hundred is what the pilot on seed 7 asked for: the hardest comparison it could settle needed 75, so the run takes the largest size the design allows
    n_repeats: int = 10  # how many seeds each stochastic planner is run with on every map. The per-map median is the paired value in the analysis and the spread across the repeats is reported as reliability. APF is deterministic and runs once whatever this says. Repeat 0 is seeded the way every run was before repeats existed, so old rows are reproduced by it
    scenarios: tuple = (
        "normal_city",
        "blocked_road",
        "dense_city",
    )  # the static scenario names. They must match the keys registered in maps/scenarios.py

    # ======================
    # APF parameters
    apf_k_att: float = 1.0  # strength of the attractive force towards the goal
    apf_k_rep: float = 120.0  # strength of repulsive force away from obstacles. Kept by the seed 11 tuning sweep on 2026-09-15 from (60, 120, 240)
    apf_rho0: float = 3.0  # Cut-off distance of repulsive force from objects. Tuned to fit streets so there are driveable channels in road sections with obstacles on other side. Large rho may make the buildings on either side push too strongle and freeze APF. Kept by the seed 11 tuning sweep on 2026-09-15 from (2.0, 3.0, 4.5)
    apf_step: float = 0.7  # the number of cells the vehicle moves each step. Kept by the seed 11 tuning sweep on 2026-09-15 from (0.5, 0.7, 1.0)
    apf_max_iter: int = (
        4000  # give up after this many steps (preventing infinite livelock loops)
    )

    # =======================
    # RRT* parameters
    rrt_step: float = 4.0  # the step distance when adding new branches. Kept by the seed 11 tuning sweep on 2026-09-15 from (3, 4, 6)
    rrt_max_iter: int = 6000  # maximum iterations or branches before giving up (dense mazes would need more samples)
    rrt_goal_bias: float = 0.1  # 10% of the time, the random point is set at the goal. Kept by the seed 11 tuning sweep on 2026-09-15 from (0.05, 0.1, 0.2)
    rrt_radius: float = 12.0  # neighbours in this distance around the node are considered for rewiring. Chosen by the seed 11 tuning sweep on 2026-09-15 from (6, 8, 12): every value solved every map and 12 drove fastest, at the cost of more neighbour checks per sample
    rrt_goal_thresh: float = (
        4.0  # if a branch lands within this of the goal connect it to the goal
    )

    # =======================
    # Hybrid parameters
    stuck_window_fixed: int = 8  # the non-adaptive hybrid calls the vehicle stuck after this many consecutive steps without progress. Chosen by the seed 11 tuning sweep on 2026-09-15 from (8, 12, 18): every value solved every map and 8 drove fastest
    stuck_delta: float = 3.0  # displacement below this many cells across the window counts as no progress. Chosen by the seed 11 tuning sweep on 2026-09-15 from (1.5, 2.0, 3.0): every value solved every map and 3.0 drove fastest. It has to sit well above apf_step: a vehicle bouncing between two points a single step apart covers that step every iteration, so a threshold below the step size can never fire on the oscillation it is meant to catch. Over a window of a dozen steps a vehicle that is genuinely getting anywhere covers several cells, so this is still a low bar
    subgoal_dist: float = 18.0  # the RRT* escape aims this far along the straight line towards the goal instead of at the goal itself. Before a sub-goal is accepted the line from it towards the goal is checked for this same distance again, and a blocked line means the sub-goal is in a pocket. Kept by the seed 11 tuning sweep on 2026-09-15 from (12, 18, 27)
    subgoal_max_doublings: int = 3  # how many times the sub-goal distance may double when the line ahead of the sub-goal is blocked. Three doublings reach eight times the distance, which is further than any map here, so the last try is effectively the goal itself
    max_escapes: int = 12  # safety cap on the number of RRT* escapes in one run, which stops a run bouncing between modes forever

    # =======================
    # Adaptive stuck detection parameters
    stuck_window_min: int = (
        4  # the smallest window, used where the surroundings are packed with obstacles. Kept by the seed 11 tuning sweep on 2026-09-15 from (3, 4, 6): every value gave identical rows, the adaptive window never changed an outcome
    )
    stuck_window_max: int = 20  # the largest window, used in open space. The hysteresis deque is capped at this value so it must be the larger of the two
    density_radius: int = (
        6  # radius in cells of the circle the local obstacle density is measured over. Kept by the seed 11 tuning sweep on 2026-09-15 from (4, 6, 9): every value gave identical rows, the adaptive window never changed an outcome
    )

    # =======================
    # Smoothing parameters
    turn_threshold: float = 0.35  # a vertex counts as a sharp corner once the heading changes by more than this many radians (about 20 degrees). APF steps bend far less than this so they are left alone
    smooth_offset: int = 6  # how many path points on each side of a run of corners are pulled into the smoothing window. Chosen by the seed 11 tuning sweep on 2026-09-15 from (3, 4, 6): every value solved every map and 6 drove fastest
    smooth_samples: int = 24  # how many points are sampled along each replacement curve
    dubins_radius: float = 2.5  # minimum turning radius in cells for the Dubins smoother. Smaller radii fit tighter streets but model a less realistic vehicle
    vehicle_radius: float = 1.0  # half-width of the vehicle in cells. Every obstacle is grown by this much when a map is built, so the planners and the scorer see the map the vehicle's body has to fit through, and the moving obstacle uses it in its overlap test. It describes the same vehicle as dubins_radius
    dubins_step: float = (
        0.4  # arc length in cells between samples when a Dubins curve is reconstructed
    )

    # =======================
    # Dynamic scenario parameters
    dynamic_obstacle_half: int = (
        4  # the moving obstacle is a square of (2 * half + 1) cells on each side, grown with the corridor when the map went from sixty to ninety cells
    )
    dynamic_speed_factors: tuple = (0.5, 1.0, 1.5)  # the obstacle's speed as a multiple of the vehicle's, one level per study. The vehicle covers apf_step cells per time step, so the obstacle covers apf_step times the factor. Slower than the vehicle, a driver that reacts can get round it; faster, a route planned around where it stands is caught when it moves
    dynamic_instances: int = 100  # how many obstacle walks the dynamic study runs, each met at every speed by every driver
    dynamic_corridor_half: int = 13  # half the height of the corridor in cells. The corridor is kept narrow enough that the obstacle blocks a real share of it, otherwise the two would rarely meet at all
    dynamic_max_steps: int = (
        1500  # give up on the dynamic run after this many time steps
    )
    replan_horizon: int = 10  # a replan driver plans again when the next this many steps of its route fall within reach of the obstacle, where reach is how far the obstacle can travel in that many steps. It also waits this many steps after a plan before it will plan again, since the new route already goes around where the obstacle stands
    replan_period: int = 0  # a replan driver also plans again every this many steps regardless of the obstacle. 0 turns the period off, so only the threat check replans
    dynamic_rho0: float = 10.0  # influence radius of the moving obstacle. It is much wider than apf_rho0 so the vehicle gives way early instead of driving up to the obstacle before reacting, but it stays under the width of the corridor so the obstacle cannot push the vehicle into the far wall from across the channel
    dynamic_k_rep_scale: float = 3.0  # the moving obstacle repels this many times harder than a static one, because being hit ends the run

    # =======================
    # Mission time parameters
    lateral_accel: float = 2.0  # the sideways acceleration the vehicle will tolerate in a turn, in cells per second squared with one cell as one metre. It sets how fast a turn of a given radius can be taken: speed is the square root of this times the radius, capped at vehicle_speed. Two is a cautious figure for a laden vehicle on broken ground
    vehicle_speed: float = 8.3  # how many cells the vehicle covers per second of driving. One cell is taken as one metre, so this is 30 km/h, which is a realistic average for an emergency vehicle picking its way through a damaged street network rather than running clear roads. Mission time is planning time plus distance over this speed, so changing it changes how much the planning cost matters against the driving cost

    # =======================
    # Output parameters
    out_dir: Path = field(
        default_factory=lambda: Path("outputs")
    )  # where the CSVs, figures and animations are written

    def __post_init__(self):
        # Accept a plain string for the output directory and make sure the folder exists so no later save can fail
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
