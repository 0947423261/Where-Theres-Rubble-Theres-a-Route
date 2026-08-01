"""
utils/optimal.py

The shortest route on the grid, used as the yardstick every planner's route length is divided by.

A* over the cells, eight-connected with Euclidean step costs, run once per map because it is deterministic. It uses the same segment rule as the planners, so a diagonal step between two free cells is refused when the sampled line between them clips an obstacle, and it stops on any cell within the goal tolerance since that is where the planners are allowed to stop too. The result is the length a perfect planner restricted to grid moves would drive, and length_ratio in the results is the planner's length over this.
"""

import heapq

import numpy as np

from .collision import point_blocked, segment_blocked

# The eight moves and what each costs
MOVES = (
    ((1, 0), 1.0),
    ((-1, 0), 1.0),
    ((0, 1), 1.0),
    ((0, -1), 1.0),
    ((1, 1), np.sqrt(2.0)),
    ((1, -1), np.sqrt(2.0)),
    ((-1, 1), np.sqrt(2.0)),
    ((-1, -1), np.sqrt(2.0)),
)


def astar_length(grid, start, goal, goal_tolerance):
    """
    Length of the shortest eight-connected route from the start cell to any cell within goal_tolerance of the goal, or None when no route exists.
    """
    h, w = grid.shape

    start_x, start_y = int(round(start[0])), int(round(start[1]))
    goal_x, goal_y = float(goal[0]), float(goal[1])

    # Nothing to search from a buried start
    if point_blocked(grid, start_x, start_y):
        return None

    def heuristic(x, y):
        # Straight-line distance less the tolerance, which never overestimates the distance to the nearest cell that counts as arrived
        return max(0.0, np.hypot(x - goal_x, y - goal_y) - goal_tolerance)

    # Cheapest known cost to each cell, and the open set ordered by cost plus heuristic
    best = {(start_x, start_y): 0.0}
    frontier = [(heuristic(start_x, start_y), 0.0, start_x, start_y)]
    closed = set()

    while frontier:
        _, cost, x, y = heapq.heappop(frontier)

        # A cell can be pushed more than once; only the first pop, which is the cheapest, is expanded
        if (x, y) in closed:
            continue
        closed.add((x, y))

        # Arrived, by the same rule the planners use
        if np.hypot(x - goal_x, y - goal_y) <= goal_tolerance:
            return float(cost)

        for (change_x, change_y), step in MOVES:
            next_x, next_y = x + change_x, y + change_y

            if not (0 <= next_x < w and 0 <= next_y < h) or (next_x, next_y) in closed:
                continue

            # The move has to pass the same sampled segment test the planners' steps pass, so a diagonal cannot cut a corner they could not
            if segment_blocked(grid, (x, y), (next_x, next_y)):
                continue

            next_cost = cost + step
            if next_cost < best.get((next_x, next_y), float("inf")):
                best[(next_x, next_y)] = next_cost
                heapq.heappush(frontier, (next_cost + heuristic(next_x, next_y), next_cost, next_x, next_y))

    return None
