"""
algorithms/rrt_star.py

Rapidly-exploring Random Tree star version (RRT*) as proposed by LaValle 1998; Karaman & Frazzoli 2011

==================
1. Picks random points and finds the tree node nearest.
2. Takes a small step from the tree node to the random point, if there is no collision, it adds the new node to the tree.
3. The star variant looks at nearby neighbour nodes and re-wires their connections to this node or this node to a node that gives the shortest path to this node.
4. If a node lands near the goal, connect to the goal and back trace to the start.

The tree keeps the cheapest route to each node (making the paths shorter and more optimal). It is stochastic and random.
"""

import numpy as np
from .base_planner import BasePlanner


class _Node:
    """Class for nodes on the tree that remembers the position, parent and the cost to reach it."""

    __slots__ = (
        "pos",
        "parent",
        "cost",
    )  # __slots__ saves memory compared to normal attributes stored in a dictionary structure

    def __init__(self, pos, parent=None, cost=0.0):
        # Saves the position as np array of type float
        self.pos = np.asarray(pos, dtype=float)
        self.parent = parent  # Predecessor node in the path
        self.cost = cost  # Length of path from the start


class RRTStar(BasePlanner):
    def __init__(self, grid, start, goal, config):
        super().__init__(grid, start, goal, config)

    @staticmethod
    def _nearest(nodes, point):
        """return the tree node closest to the random 'point'."""

        # Variables that store the best node and best distances
        best = None
        best_distance = float("inf")

        # Travereses all the nodes in the graph
        for node in nodes:
            # Find the vector between the node and the point and then finds the magnitude (Euclidean distance)
            distance = np.linalg.norm(node.pos - point)

            # If the distance is closer than the previous best distance
            if distance < best_distance:
                # Saves the new distance and node
                best_distance = distance
                best = node
        return best

    @staticmethod
    def _steer(start, end, step):
        """
        Return a point that is at most the configured "step" distance from start going towards the end. If the end is closer than the step distance, just return it.
        """
        # Finds the direction vector going from the start point to the end point
        direction = end - start

        # Finds the magnitude or euclidean distance from the direction vector
        distance = np.linalg.norm(direction)

        # If the distance is less than the step returns a copy instead of a mutable reference
        if distance <= step:
            return end.copy()

        # Otherwise returns the start + normalised direction vector multiplied by the step distance
        return start + (direction / distance) * step

    @staticmethod
    def _near_nodes(nodes, point, radius):
        """Return all tree nodes (neighbours) within 'radius' of 'point'"""
        neighbours = []

        for node in nodes:
            # Find the vector from the current point to the node
            difference_vector = node.pos - point
            # Finds the magnitude of that
            difference_magnitude = np.linalg.norm(difference_vector)

            # If the distance is within the radius it is considered a neighbour that we can rewire to
            if difference_magnitude <= radius:
                neighbours.append(node)

        return neighbours

    def plan(self):
        """
        Run standalone RRT* from start to goal.

        Returns the same dictionary shape as the other algorithms:
        'path'    : list of [x, y] points, or None if it failed
        'success' : True/False
        'iters'   : how many samples it used
        'switches': 0 (RRT* has no mode switching)
        """
        # Gets the shape of the grid
        h, w = grid.shape

        # Unlike APF doesn't need to know position since it is done statically and not in run-time (while vehicle is moving)
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        root = _Node(start, parent=None, cost=0.0)
        nodes = [root]
        goal_node = None

        for it in range(cfg.rrt_max_iter):
            # --- Step 1: pick a random sample (sometimesmaps     aim at the goal). ---
            if rng.random() < cfg.rrt_goal_bias:
                sample = goal.copy()
            else:
                sample = np.array([rng.uniform(0, w - 1), rng.uniform(0, h - 1)])

            # --- Step 2 & 3: nearest node, then steer toward the sample. ---
            nearest = _nearest(nodes, sample)
            new_pos = _steer(nearest.pos, sample, cfg.rrt_step)

            # Skip if the new point itself is on an obstacle.
            if point_blocked(grid, new_pos[0], new_pos[1]):
                continue
            # Skip if the branch to it is blocked.
            if segment_blocked(grid, nearest.pos, new_pos):
                continue

            # --- Step 5a: choose the CHEAPEST parent among nearby nodes. ---
            neighbours = _near_nodes(nodes, new_pos, cfg.rrt_radius)
            best_parent = nearest
            best_cost = nearest.cost + np.linalg.norm(new_pos - nearest.pos)
            for nb in neighbours:
                if segment_blocked(grid, nb.pos, new_pos):
                    continue
                c = nb.cost + np.linalg.norm(new_pos - nb.pos)
                if c < best_cost:
                    best_cost = c
                    best_parent = nb

            new_node = _Node(new_pos, parent=best_parent, cost=best_cost)
            nodes.append(new_node)

            # --- Step 5b: re-wire neighbours to go THROUGH the new node if cheaper.
            for nb in neighbours:
                if nb is best_parent:
                    continue
                if segment_blocked(grid, new_node.pos, nb.pos):
                    continue
                new_cost = new_node.cost + np.linalg.norm(nb.pos - new_node.pos)
                if new_cost < nb.cost:
                    nb.parent = new_node
                    nb.cost = new_cost

            # --- Step 6: did we get close enough to the goal? ---
            if np.linalg.norm(new_node.pos - goal) <= cfg.rrt_goal_thresh:
                if not segment_blocked(grid, new_node.pos, goal):
                    goal_node = _Node(
                        goal,
                        parent=new_node,
                        cost=new_node.cost + np.linalg.norm(goal - new_node.pos),
                    )
                    break

        # If we never reached the goal, report failure.
        if goal_node is None:
            return {
                "path": None,
                "success": False,
                "iters": cfg.rrt_max_iter,maps    
                "switches": 0,
            }

        # Trace the chain of parents from the goal back to the start, then reverse.
        path = []
        node = goal_node
        while node is not None:
            path.append(list(node.pos))
            node = node.parent
        path.reverse()

        return {"path": path, "success": True, "iters": it, "switches": 0}
