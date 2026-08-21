"""
utils/visualise.py

Draws a map with a path on top of it and saves it as a picture.

Every figure and every animation in the project renders its map through render_grid, so the colours mean the same thing everywhere: light grey is a driveable street, dark slate is an intact building, brown is rubble, amber is the moving obstacle and a faint grey ring around each obstacle is the margin the vehicle's body cannot enter. Buildings are drawn at their true size; the margin is the planner's view of them.

Matplotlib is switched to the Agg backend, which draws into a file instead of onto a screen. That keeps the figures working over a terminal session with no display attached.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# The shared colour scheme, indexed by cell value: 0 street, 1 building, 2 rubble, 3 moving obstacle, 4 vehicle margin
GRID_CMAP = ListedColormap(["#f7f8fa", "#3b4048", "#a9714b", "#e9a13b", "#e3e6eb"])

# Colours used for the route, the two ends of it and the mode switches, kept here so the figures and the animations match
PATH_COLOUR = "#2f6fb3"
START_COLOUR = "#2a9d8f"
GOAL_COLOUR = "#b3402f"
JUNCTION_COLOUR = "#e9a13b"


def render_grid(ax, grid):
    """
    Draw the map onto an existing matplotlib axis using the shared colours.

    The colour scale is pinned between 0 and 4 so a map with no rubble still draws its buildings in the building colour, and origin="lower" puts cell (0, 0) at the bottom left so the picture matches the (x, y) convention the planners use.
    """
    ax.imshow(
        grid,
        cmap=GRID_CMAP,
        vmin=0,
        vmax=4,
        origin="lower",
        interpolation="nearest",
    )


def mark_ends(ax, start, goal):
    """Draw the start as a square and the goal as a star, which is how they appear in every figure."""
    ax.scatter(
        [start[0]], [start[1]], c=START_COLOUR, s=90, marker="s", zorder=4, label="start"
    )
    ax.scatter(
        [goal[0]], [goal[1]], c=GOAL_COLOUR, s=140, marker="*", zorder=4, label="goal"
    )


def draw_path(grid, start, goal, path, title, save_path, junction_points=None, extra_paths=None):
    """
    Draw one map with one path across it and write it to save_path.

    grid        the 2D map
    start, goal the two ends as (x, y)
    path        the list of points from a planner result, which may be None when the planner failed
    title       the caption above the picture
    save_path   the file to write, where the extension decides the format
    junction_points optional list of (x, y) points where the hybrid switched to RRT*, drawn as amber stars
    extra_paths optional list of (path, label, colour) to overlay a second route on the same map
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # The map itself, in the shared colours
    render_grid(ax, grid)

    # The main route, drawn only when the planner produced one
    if path is not None and len(path) > 1:
        points = np.asarray(path, dtype=float)
        ax.plot(
            points[:, 0], points[:, 1], color=PATH_COLOUR, linewidth=2.2, zorder=3, label="path"
        )

    # Any extra routes, used to put two algorithms on one figure
    if extra_paths:
        for extra_path, label, colour in extra_paths:
            if extra_path is not None and len(extra_path) > 1:
                points = np.asarray(extra_path, dtype=float)
                ax.plot(
                    points[:, 0], points[:, 1], color=colour, linewidth=1.8, zorder=2, label=label
                )

    # The points where the hybrid handed control to RRT*, so a reader can see where the escapes happened
    if junction_points:
        junction_points = np.asarray(junction_points, dtype=float)
        if len(junction_points):
            ax.scatter(
                junction_points[:, 0],
                junction_points[:, 1],
                c=JUNCTION_COLOUR,
                s=70,
                marker="*",
                zorder=5,
                label="RRT* escape",
            )

    # The two ends of the route
    mark_ends(ax, start, goal)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_xlim(-1, grid.shape[1])
    ax.set_ylim(-1, grid.shape[0])

    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")

    # Close the figure to release its memory, which matters when a run saves dozens of them
    plt.close(fig)

    return save_path
