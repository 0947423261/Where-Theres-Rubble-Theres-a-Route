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
    # APF parameters
    apf_k_att: float = 1.0  # strength of the attractive force towards the goal
    apf_k_rep: float = 120.0  # strength of repulsive force away from obstacles
    apf_rho0: float = 3.0  # Cut-off distance of repulsive force from objects. Tuned to fit streets so there are driveable channels in road sections with obstacles on other side. Large rho may make the buildings on either side push too strongle and freeze APF
    apf_step: float = 0.7  # the number of cells the vehicle moves each step
    apf_max_iter: int = (
        4000  # give up after this many steps (preventing infinite livelock loops)
    )

    # =======================
    # RRT* parameters
    rrt_step: float = 4.0  # the step distance when adding new branches
    rrt_max_iter: int = 6000  # maximum iterations or branches before giving up (dense mazes would need more samples)
    rrt_goal_bias: float = 0.1  # 10% of the time, the random point is set at the goal
    rrt_radius: float = (
        8.0  # neighbours in this distance around the node are considered for rewiring
    )
    rrt_goal_thresh: float = (
        4.0  # if a branch lands within this of the goal connect it to the goal
    )
