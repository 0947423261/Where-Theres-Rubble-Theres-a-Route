"""
algorithms/rrt_star.py

Rapidly-exploring Random Tree star version (RRT*) as proposed by LaValle 1998; Karaman & Frazzoli 2011

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
    def __init__(self, grid, start, goal, config, rng, trace=False):
        super().__init__(grid, start, goal, config, rng)

        # Whether to record how the tree grew, for the animation. Off in the experiment so nothing extra is done inside the timed run. Recording touches no random draw and no arithmetic, so the route is the same either way
        self.trace = trace

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

        # The tree always holds at least the root, so this cannot happen. If it ever does, fail loudly rather than silently killing the process
        if best is None:
            raise ValueError("RRT* asked for the nearest node of an empty tree")

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
    def _neighbours(nodes, point, radius):
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
        'trace'   : only when tracing, the tree's history as a list of events: ("add", node, parent) when a node joins, ("rewire", node, old parent, new parent) when a neighbour changes parent, ("goal", goal, parent) when the goal connects. Positions are (x, y) tuples
        """
        # Gets the shape of the grid
        h, w = self.grid.shape

        # Unlike APF doesn't need to know position since it is done statically and not in run-time (while vehicle is moving)
        # Converts the start and goal to numpy array format as float type
        start = np.asarray(self.start, dtype=float)
        goal = np.asarray(self.goal, dtype=float)

        ### === Initialisation === ###
        # Creates the root node at the start with 0 cost
        root = _Node(start, parent=None, cost=0.0)
        # Adds root node to list of node
        nodes = [root]
        # Sets goal node as none
        goal_node = None

        # The tree's history, only kept when tracing
        events = []
        ### === END ===

        # Iterates up until we reach the maximum configured number of iterations for RRT
        for iteration in range(self.config.rrt_max_iter):
            # Generates a random floating point between 0 and 1  if it is less than the goal bias (percentage of times we select the goal), then the point is the goal
            if self.rng.random() < self.config.rrt_goal_bias:
                sample = goal.copy()
            else:
                # Otherwise the point is a random point within the boundary of the map
                sample = np.array(
                    [self.rng.uniform(0, w - 1), self.rng.uniform(0, h - 1)]
                )

            # Finds nearest node to the sample point
            nearest = self._nearest(nodes, sample)

            # Finds the point along the line from sample to nearest node to add the new node
            new_pos = self._steer(nearest.pos, sample, self.config.rrt_step)

            # If the branch from the nearest node to the new node passes through an obstacle or leaves the map, the new node can't be added. The sampled check covers the new node itself since the endpoints are sampled too. Without this the tree grows through buildings and the traced path is not driveable
            if self._segment_blocked(nearest.pos, new_pos):
                continue

            # Finds the neighbours
            neighbours = self._neighbours(nodes, new_pos, self.config.rrt_radius)

            # === Finding the Best Parent ===
            # RRT* doesn't use the nearest node as the parent but rather the node that is within the radius with smallest cost to start
            # Initialise with best parent being the nearest node
            best_parent = nearest
            # The initial cost is the cost to the nearest node + the distance from the parent to the new node
            best_cost = nearest.cost + np.linalg.norm(new_pos - nearest.pos)

            for neighbour in neighbours:
                # If the path between the neighbour and the node is blocked by an obstacle we can't make that neighbour the parent and connect to it
                if self._segment_blocked(neighbour.pos, new_pos):
                    continue

                # Finds the cost if we use the neighbour as the parent
                cost = neighbour.cost + np.linalg.norm(new_pos - neighbour.pos)
                # If the cost beats the current best we will use this neighbour as the parent for now (set it as best)
                if cost < best_cost:
                    best_cost = cost
                    best_parent = neighbour

            # Creates a new node at the new node position, with cost set as the cost using whatever was determined to be the parent
            new_node = _Node(new_pos, parent=best_parent, cost=best_cost)
            # Add the node to list of nodes
            nodes.append(new_node)

            if self.trace:
                events.append(("add", tuple(new_node.pos), tuple(best_parent.pos)))
            ### === END ===

            ### === Rewiring ===
            for neighbour in neighbours:
                # If the neighbour is the parent we can't rewire the parent to use this node as its parent (loop)
                if neighbour is best_parent:
                    continue
                # If the path between the new node and the neighbour is blocked by an obstacle we can't rewire the neighbour to use this node
                if self._segment_blocked(new_node.pos, neighbour.pos):
                    continue
                # Sees what the cost would be for the neighbour to potentially rewire would be if it used this node as a connection instead
                new_cost = new_node.cost + np.linalg.norm(neighbour.pos - new_node.pos)

                # If the new cost beats the current cost for the neighbour we rewire the neighbour to use this node and have its new cost
                if new_cost < neighbour.cost:
                    if self.trace:
                        events.append(("rewire", tuple(neighbour.pos), tuple(neighbour.parent.pos), tuple(new_node.pos)))
                    neighbour.parent = new_node
                    neighbour.cost = new_cost
            ### === END ===

            vector_to_goal = new_node.pos - goal
            # Checks if the distance to the goal is less than the configured threshold distance for reaching the goal
            if np.linalg.norm(vector_to_goal) <= self.config.rrt_goal_thresh:
                # If there is no obstacle between the node and the goal, we set the parent of the goal node to be the new node and the cost as the new node cost + distance to the goal node
                if not self._segment_blocked(new_node.pos, goal):
                    goal_node = _Node(
                        goal,
                        parent=new_node,
                        cost=new_node.cost + np.linalg.norm(goal - new_node.pos),
                    )
                    if self.trace:
                        events.append(("goal", tuple(goal), tuple(new_node.pos)))
                    break

        # If goal node is not reached report failure
        if goal_node is None:
            result = {
                "path": None,
                "success": False,
                "iters": self.config.rrt_max_iter,
                "switches": 0,
            }
            if self.trace:
                result["trace"] = events
            return result

        # Traceback the path from the goal node
        path = []
        node = goal_node
        while node is not None:
            # Add each node position to the path
            path.append(list(node.pos))
            # Traverse to the parent node
            node = node.parent

        # Reverse the path to go from start to goal
        path.reverse()

        result = {"path": path, "success": True, "iters": iteration, "switches": 0}
        if self.trace:
            result["trace"] = events
        return result
