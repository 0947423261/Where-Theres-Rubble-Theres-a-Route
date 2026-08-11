"""
algorithms/smoothers.py

Path-smoothing strategies for the hybrid planner. We use a hierarchy to implement the staregy pattern where a common smoother interface hides the exact smoother instance. Every smoother returns a path that is collision-free.

The smoothing happens at the junction point of mode switch and every node that is part of the RRT* escape since RRT* has jagged edges whereas APF is relatively straight. The smoothing just smoothens sharp turns, the near-straight APF steps fall below the threshold and are ignored

There are many ways to smoothen, this includes:
- Bezier (simple)
- B-spline (more smooth)
- Dubins (guarantees curvature that is drivable)
"""

from abc import ABC, abstractmethod
import math

import numpy as np
from scipy.interpolate import BSpline
from ..utils.collision import validate_path

# Full turn (360 degree) constant
TAU = 2.0 * math.pi

def _mod2pi(theta):
    """Modulus operation to make sure any angle goes into the range [0, 2*pi) where it is equivalent when referring to steering angles."""
    return theta - TAU * math.floor(theta / TAU)


def _angle_gap(a, b):
    """Smallest absolute difference between two headings, in between [0, pi]."""
    diff = (a - b + math.pi) % TAU - math.pi
    return abs(diff)


# Dubins puts realistic constraints on the vehicle: it can't turn on the spot, it can only move forwards, it has a minimum turning radius
# ---------------------------------------------------------------------------
# The six Dubins words: left -> straight -> left (LSL), right -> straight -> right (RSR) etc etc. Dubins proved one of these sequences will be optimal. 
# Each takes the normalised inputs (alpha, beta, d) 
# Where:
# "a" is the initial heading after transforming the problem in a normalised coordinate system
# "b" is the final heading in the normalised system
# "d" is the distance between the start and goal divided by the turning radis

# The functions return:
# "t" being the turning angle (arc-length) for the first action in (RSR for example) 
# "p" being the distance to drive straight or the turning angle/arc length for the middle circle
# "q" being the turning angle (arc-length) for the last action

# angles are in radians
# These are the standard closed-form equations (Shkel & Lumelsky, 2001).
# ---------------------------------------------------------------------------

"""
ADD DIAGRAMS TO DISS FOR THIS
"""

def _dubins_LSL(a, b, d):
    # Shorthands for sin and cos of the angles and the difference angle between initial and final heading angle
    sa, sb, ca, cb, cab = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)
    # Compute the straight segment length connecting the two turning arcs 
    p_sq = 2 + d * d - 2 * cab + 2 * d * (sa - sb)
    # If the squared distance is less than 0 it is not a possible path
    if p_sq < 0:
        return None

    # Computes the tangent direction that connects the two turning arcs which is the same as the angle connecting the centers of the turning circles
    tangent_direction = math.atan2(cb - ca, d + sa - sb)

    # The first angle is the angle that the vehicle turns until it aligns with the tangent angle
    # The second return is the distance travelled 
    # The third return is the angle the vehicle turns to achieve the final heading angle from the tangent direction
    return _mod2pi(-a + tangent_direction), math.sqrt(p_sq), _mod2pi(b - tangent_direction), ("L", "S", "L")

# follow similar explanations to that above
def _dubins_RSR(a, b, d):
    sa, sb, ca, cb, cab = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)
    p_sq = 2 + d * d - 2 * cab + 2 * d * (sb - sa)
    if p_sq < 0:
        return None
    tangent_direction = math.atan2(ca - cb, d - sa + sb)
    return _mod2pi(a - tangent_direction), math.sqrt(p_sq), _mod2pi(-b + tangent_direction), ("R", "S", "R")


def _dubins_LSR(a, b, d):
    sa, sb, ca, cb, cab = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)
    p_sq = -2 + d * d + 2 * cab + 2 * d * (sa + sb)
    if p_sq < 0:
        return None
    p = math.sqrt(p_sq)
    tangent_direction = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
    return _mod2pi(-a + tangent_direction), p, _mod2pi(-b + tangent_direction), ("L", "S", "R")


def _dubins_RSL(a, b, d):
    sa, sb, ca, cb, cab = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)
    p_sq = d * d - 2 + 2 * cab - 2 * d * (sa + sb)
    if p_sq < 0:
        return None
    p = math.sqrt(p_sq)
    tangent_direction = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
    return _mod2pi(a - tangent_direction), p, _mod2pi(b - tangent_direction), ("R", "S", "L")

# Slightly different since there is no straight line connecting the turns, more optimal for short distance maneuver where the start and end are too close to allow for an outer or inner tangent
def _dubins_RLR(a, b, d):
    sa, sb, ca, cb, cab = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)

    # The three turning circles must be mutually tangential so the centers form an isosceles triangle. This is the cosine of the angle related to the middle turn
    cos_middle = (6.0 - d * d + 2 * cab + 2 * d * (sa - sb)) / 8.0

    # If the distance between the center of the first and third circle is greater than 4R where R is the distance from the center of one of the circles to the center of the second circle then the middle circle can't touch both to bridge the gap
    if abs(cos_middle) > 1.0:
        return None

    # This is the ravel distance along the middle circle arc
    p = _mod2pi(TAU - math.acos(cos_middle))

    # The vehicle turns around a first circle from the initial heading. T is the distance travelled along the first circle arc
    t = _mod2pi(a - math.atan2(ca - cb, d - sa + sb) + p / 2.0)

    # The vehicle makes the final turn towards the final heading, q is the travel distance over the final circle arc
    q = _mod2pi(a - b - t + p)
    return t, p, q, ("R", "L", "R")


# Similar explanation to above
def _dubins_LRL(a, b, d):
    sa, sb, ca, cb, cab = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)
    cos_middle = (6.0 - d * d + 2 * cab + 2 * d * (sb - sa)) / 8.0

    if abs(cos_middle) > 1.0:
        return None

    p = _mod2pi(TAU - math.acos(cos_middle))
    t = _mod2pi(-a - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
    q = _mod2pi(_mod2pi(b) - a - t + p)
    return t, p, q, ("L", "R", "L")

# collection of all six dubin word functions
_DUBINS_WORDS = (_dubins_LSL, _dubins_RSR, _dubins_LSR, _dubins_RSL, _dubins_RLR, _dubins_LRL)


class Smoother(ABC):
    """Abstract base for all path-smoothing strategies."""

    # Constructor just holds the configuration 
    def __init__(self, config):
        self.config = config

    def smooth(self, path, grid):
        """
        Return a new path with the sharp corners softened. Default behaviour is find sharp corners and rounds them using the desired smoother. 'NoSmoother' overrides this to do nothing.
        """
        return self._round_corners(path, grid)

    @abstractmethod
    def _curve(self, window, grid):
        """
        Transforms one corner window (a list of points, the corner at its centre) into a replacement curve that starts at window[0] and ends at window[-1]. Returns the list of points or None if not possible.
        """
        raise NotImplementedError


    # In diss make a diagram for this
    def _round_corners(self, path, grid):
        """
        Finds the sharp corners, groups any that are consecutive into a single run (a jagged RRT* stretch is one run, not many separate corners), and replaces each run with the strategy's '_curve' implementation. It keeps original if the curve collides.
        """

        # Gets each point as an numpy array of type float in case it isn't 
        points = [np.asarray(p, dtype=float) for p in path]

        # If there are less than 3 points, it is too short to be a corner
        if len(points) < 3:
            return points

        # The offset is the configured number of points to include on each side of the window 
        offset = self.config.smooth_offset

        # Finds a list of corners where the path turns sharply 
        corners = self._find_corners(points, self.config.turn_threshold)

        # Group consecutive corner into a run to smoothen zig-zags 
        runs = self._group_runs(corners)

        #  Tracks the left most index that has already been smoothed to prevent overlapping windows from interfering. The algorithm starts from the end of the path backwards so having it set as len(points) means we haven't started moving in from the left and smoothing
        processed_leftmost = len(points)

        # Sort the run by their first index in descending order so it starts from the end of the path to the start. This is because if we go left to right since smoothing can change list length it would mess up transformations that occur more to the right in the list
        for run in sorted(runs, key=lambda r: r[0], reverse=True):


            # Get the indices around the run by offset points on each side (each boundary) to get some points outside where the corner starts to smooth and turn into. The min and max prevent it from going out of bounds of the run
            left_boundary= max(0, run[0] - offset)
            right_boundary = min(len(points) - 1, run[-1] + offset)

            # Need at least three points to form a curve
            if right_boundary - left_boundary< 2:
                continue

            # If the window runs into an already smoothed region don't smooth
            if right_boundary >= processed_leftmost:
                continue

            # Creates a slice of the path that contains the window
            window = points[left_boundary: right_boundary + 1]

            # Generates a smooth replacement curve for the window
            curve = self._curve(window, grid)

            # Ensure there is a curve and that the curve is collision free
            if curve is not None and validate_path(grid, curve):

                # If the curve is valid, replace the window points with the smoothed curve points
                points[left_boundary: right_boundary + 1] = [np.asarray(c, dtype=float) for c in curve]

                # Set the new processed edge marker to be the left-most index in the new smoothed curve
                processed_leftmost = left_boundary
            # Otherwise fall back to original sharp points 
        return points

    @staticmethod
    def _group_runs(corners):
        """
        Collapse a sorted list of corner indices into a 2D array of connected runs so connected "consecutive corners" are smoothed together.
        """

        runs = []
        for c in corners:
            # Check if the runs array isn't empty and then checks if c is one more than the last element of the run
            if runs and c == runs[-1][-1] + 1:
                # If it is, we append c into that run group
                runs[-1].append(c)
            else:
                # Otherwise append c to a new run group
                runs.append([c])
        return runs

    @staticmethod
    def _find_corners(points, threshold):
        """
        Return the indices of interior vertices whose turn angle exceeds "threshold" radians. The turn angle is the change in heading between the segment arriving at a vertex and the one leaving it. APF would be negligible and therefore ignored.
        """

        # Stores the indices where sharp corners are found
        corners = []

        # Iterate through the points in the list of points
        for index in range(1, len(points) - 1):

            # Finds the segment vectors (vector from previous point to current and vector from current point to the next)
            incoming = points[index] - points[index - 1]
            outgoing = points[index + 1] - points[index]

            # Finds the length of these segment vectors
            length_in = np.linalg.norm(incoming)
            length_out = np.linalg.norm(outgoing)

            # Skip degenerate, negligible segments
            if length_in < 1e-9 or length_out < 1e-9:
                continue

            #  Finds the angle of the incoming segment
            a_in = math.atan2(incoming[1], incoming[0])
            # Finds the angle of the outgoing segment
            a_out = math.atan2(outgoing[1], outgoing[0])

            # Checks if the difference in angles between the headings exceeds the threshold, if so mark it as a corner
            if _angle_gap(a_in, a_out) > threshold:
                corners.append(index)

        return corners

    @staticmethod
    def _bezier_curve(control_points, n_samples):
        """
        Sample a Bezier curve whose control points are the whole window, via de Casteljau's algorithm since Bernstien high-order computation can expose floating point errors and require mass compute. 


        The curve must pass through first and last point and stay in convex hull (smallest convx polygon that contains all points as vertices) of points. 

        In bezier each point affects the whole curve whereas bspline means that each point only has effect on a portion of the curve.
        """

        # Converts the control points to a numpy array of floats which they should be
        control_points = [np.asarray(p, dtype=float) for p in control_points]
        # Initialises the list that will store all points sampled on Bezier curve
        curve = []

        # n_samples are the number of points on the curve, the loop increments over equal sized fractions in the range of 0 and 1
        for fraction in np.linspace(0.0, 1.0, n_samples):
            # copies the control points as a shallow copy since a fresh copy is needed on each iteration
            points = control_points[:]

            # Continues until the points are reduced to a single point
            while len(points) > 1:
                # Finds the weighted average of the points biased towards points that are closest to the "fraction"
                points = [(1.0 - fraction) * points[i] + fraction * points[i + 1] for i in range(len(points) - 1)]

            # Appends the new converged point to the curve list of points
            curve.append(points[0])
        return curve

    @staticmethod
    def _clamped_bspline(control_points, n_samples, degree=3):
        """
        Sample a clamped B-spline using the control points passed by parameter. Being
        clamped, the curve passes through the first and last control point while staying inside the control polygon (polygon obtained by connecting the control points of a curve in order).
        """

        # Converts the control points to numpy array of floats
        control_points = np.asarray(control_points, dtype=float)

        # Finds the number of control points
        num_control_points = len(control_points)

        # The polynomial degree can't be greater than 1 less than the number of control points 
        degree = min(degree, num_control_points - 1)
        
        # If the degree is 0 or invalid just return a straight line from first to last point
        if degree < 1:
            return [control_points[0], control_points[-1]]

        # Calculate the number internal knots (where curve segments join) 
        inner = num_control_points - degree - 1

        if inner > 0 :
            # The number of points is inner + 2 (boundaries)
            num_points = inner + 2

            # Generate evenly spaced spaced points from 0 to 1 
            fractions = np.linspace(0.0, 1.0, num_points)

            # Remove the first and last element, these are  the boundaries and not internal knots
            internal = list(fractions[1:-1])
        else:
            internal = []

        # The k+1 zeros at the start and end to clamp the curve to the first and last point
        knots = np.array([0.0] * (degree + 1) + internal + [1.0] * (degree + 1))

        # This creates the spline object with the knot vector, control points and degree
        spline = BSpline(knots, control_points, degree)

        # Sample the spline with n_samples points
        curve = [] 

        for fraction in np.linspace(0.0, 1.0, n_samples):
            curve.append(spline(fraction))

        return curve

    @staticmethod
    def _dubins_reconstruct(start_pose, types, seg_lengths, radius, step):
        """
        Reconstructs a Dubin path from the absract word into actual (x,y, heading) positions. Essentially samples across arc length in way that doesn't overshoot steering
        """
        # Unpacks the starting "pose", starting point and heading angle (theta)
        x, y, th = start_pose
        poses = [(x, y, th)]

        # Maps the word or segment type to numerical turning directions
        turn_map = {"L": +1.0, "S": 0.0, "R": -1.0}

        # Iterate through each segment in the word, types is a tuple like ("L","S","L"). Seg length is the arc length for the segments 
        for segment_type, seg_length in zip(types, seg_lengths):
            # converts the segment length to actual units by multiplying by radius
            length = seg_length * radius

            # Determines the number of steps to make
            n = max(1, int(length / step))

            # Calculates step size
            ds = length / n

            # Finds the numerical turning direction for this segment type
            turn = turn_map[segment_type]

            # loop through each small step in the segment
            for _ in range(n):
                # Finds the change of heading at the midpoint of taking the step
                th_mid = th + turn * (ds / radius) * 0.5

                # Move the vehicle in the direction of the midpoint heading by the step amount
                x += math.cos(th_mid) * ds
                y += math.sin(th_mid) * ds

                # Turns by the step amount
                th += turn * (ds / radius)

                # Adds the new step in the segment
                poses.append((x, y, th))

        return poses

    @classmethod
    def _dubins_path(cls, start, end, radius, step, position_tolerance=0.75, angle_tolerance=0.30):
        """
        Finds the shortest Dubins path between the start and end pose (x, y, theta sampled to a list of [x, y] points. Every candidate word is reconstructed using sampling along the segmenets and only kept if it actually lands on the end pose with similar/right heading.

        Return None if no word reaches the end from the start
        """
        # Gets the start and end coordinates and angles
        start_x, start_y, start_theta = start
        end_x, end_y, end_theta = end

        # Finds the change in y and change in x between the two points
        change_x, change_y = end_x - start_x, end_y - start_y 

        # Finds the distance between both points
        D = math.hypot(change_x, change_y)

        # If the distance is negligible between the points ignore, the points are the same
        if D < 1e-9:
            return None

        
        # Finds the normalised distance
        normalised_distance = D / radius

        # Finds the angle from the start to the end
        theta = _mod2pi(math.atan2(change_y, change_x))

        # Finds the start heading relative to the line connecting the points
        alpha = _mod2pi(start_theta - theta)

        # Finds the end heading relative to the line connecting the points
        beta = _mod2pi(end_theta - theta)

        # Tries all of the word types, starting from the most common
        candidates = []
        for word in _DUBINS_WORDS:

            result = word(alpha, beta, normalised_distance)

            # If the word has no result move to the next
            if result is None:
                continue

            # Gets the lengths of the first, second and third segment as well as the dubin word 
            t, p, q, types = result

            # Stores the calculated total length, the word and individual segment length
            candidates.append((t + p + q, types, (t, p, q)))

        # Sorts the candidates list by total length resulting in shortest path first
        candidates.sort(key=lambda c: c[0])

        # Iterate through the sorted candidates
        for _, types, seg_lengths in candidates:
            # Reconstructs the dubin path using sampling over the arcs and returns the list of poses (path)
            poses = cls._dubins_reconstruct(start, types, seg_lengths, radius, step)

            # get the final path
            final_x, final_y, final_theta = poses[-1]
            
            # Checks if the distance from the last point in the segment to the end is less than the tolerance and that the heading angular difference is within the tolerance
            if math.hypot(final_x - end_x, final_y - end_y) <= position_tolerance and _angle_gap(final_theta, end_theta) <= angle_tolerance:
            
                curve = []

                # Create the curve using the points along the segment
                for point_x, point_y, _ in poses:
                    numpy_point = np.array([point_x,point_y], dtype=float)
                    curve.append(numpy_point)

                # Snaps the end points to be the actual start and end
                curve[0] = np.array([start_x, start_y], dtype=float)
                curve[-1] = np.array([end_x, end_y], dtype=float)

                # Returns the curve
                return curve
        return None


class NoSmoother(Smoother):
    """
    The identity strategy: return the path untouched. Simpler to use another NoSmoother object for the strategy pattern rather than an if statement
    """

    def smooth(self, path, grid):
        return path

    def _curve(self, window, grid):
        # Never called (smooth is overridden) but required by the interface
        return None


class BezierSmoother(Smoother):
    """
    Bezier corner rounding. Cheap and visually smooth but does not consider turning radius, so a rounded corner can still be too tight for non-hamonic vehicle. Fits one Bezier curve to the whole run window, so every
    control point influences the curve globally.
    """

    def _curve(self, window, grid):
        return self._bezier_curve(window, self.config.smooth_samples)


class BSplineSmoother(Smoother):
    """
    B-spline corner rounding. Smoother than Bezier and stays inside the control
    polygon of the window, so it hugs the original path more tightly. Still no
    curvature guarantee. Uses every point in the window as a control point.
    """

    def _curve(self, window, grid):
        return self._clamped_bspline(window, self.config.smooth_samples)


class DubinsSmoother(Smoother):
    """
    Dubins corner rounding. Unlike Bezier and B-spline this *guarantees* the
    curve is drivable: its curvature never exceeds 1 / radius. It connects the
    window endpoints as full poses (position + heading, headings taken from the
    incoming and outgoing segments) with the shortest curvature-constrained path.
    """

    def _curve(self, window, grid):
        # Headings from the first and last segments of the window
        start_dir = window[1] - window[0]
        end_dir = window[-1] - window[-2]
        if np.linalg.norm(start_dir) < 1e-9 or np.linalg.norm(end_dir) < 1e-9:
            return None

        start_pose = (window[0][0], window[0][1], math.atan2(start_dir[1], start_dir[0]))
        end_pose = (window[-1][0], window[-1][1], math.atan2(end_dir[1], end_dir[0]))

        return self._dubins_path(
            start_pose, end_pose, self.config.dubins_radius, self.config.dubins_step
        )
