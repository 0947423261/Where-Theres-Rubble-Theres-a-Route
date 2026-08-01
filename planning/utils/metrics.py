"""
utils/metrics.py

Contains the measurements used to score one planned path.

Every algorithm is scored by the same functions so that no algorithm can gain an advantage from the way it is measured. The measurements are:

- success, whether the vehicle reached the goal along a path that is collision free
- path length, the distance driven, where shorter means a faster response
- smoothness, how much the route turns at an average vertex; curvature, how much it turns per cell driven; max curvature, the tightest single turn on the route; the total heading change; the count of turns sharper than the smoothing threshold; and whether the route somewhere turns tighter than the Dubins radius allows. Lower means easier to drive
- clearance, how far the route stays from obstacles, where higher means safer

Computation time is the fifth measurement and it is timed around the call to the planner in the runner, so there is nothing to compute for it here.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from ..maps.inflate import MARGIN_VALUE
from .collision import validate_path


def path_length(path):
    """
    Total distance driven along the path, found by adding the straight-line distance between each consecutive pair of points.
    """
    # A path of fewer than two points has no length
    if path is None or len(path) < 2:
        return 0.0

    total = 0.0

    for i in range(len(path) - 1):
        # Convert both ends of the segment in case the path holds plain lists
        point0 = np.asarray(path[i], dtype=float)
        point1 = np.asarray(path[i + 1], dtype=float)

        # Add the magnitude of the vector between them
        total += np.linalg.norm(point1 - point0)

    return float(total)


def total_turning(path):
    """
    The sum of every heading change along the path, in radians. This is the raw quantity both of the turning measures below are built from.
    """
    # Fewer than three points means there is no interior vertex to turn at
    if path is None or len(path) < 3:
        return 0.0, 0

    total_angle = 0.0
    count = 0

    for i in range(1, len(path) - 1):
        # The three consecutive points that form this turn
        before = np.asarray(path[i - 1], dtype=float)
        here = np.asarray(path[i], dtype=float)
        after = np.asarray(path[i + 1], dtype=float)

        # The segment vectors arriving at and leaving the middle point
        incoming = here - before
        outgoing = after - here

        # Their lengths, needed to normalise the dot product
        length_in = np.linalg.norm(incoming)
        length_out = np.linalg.norm(outgoing)

        # Skip degenerate segments where two points sit on top of each other, since they have no direction
        if length_in < 1e-9 or length_out < 1e-9:
            continue

        # The cosine of the angle between the two segments, clipped because floating point error can push it just outside the domain of arccos
        cos_angle = np.clip(
            np.dot(incoming, outgoing) / (length_in * length_out), -1.0, 1.0
        )

        # 0 radians is driving straight on and pi radians is reversing direction
        total_angle += np.arccos(cos_angle)
        count += 1

    return total_angle, count


def smoothness(path):
    """
    A value between 0 and 1 describing how much the route turns at an average vertex, where 0 is a straight line and 1 is a path that doubles back on itself at every point. Lower is better.

    Read this one with care. It divides the total turning by the number of vertices, so it measures the turn at an average vertex rather than the shape of the route. Two planners that drive the same corner score differently when one samples it with six points and the other with one, and APF samples far more finely than RRT* does. Use curvature below when the comparison is between algorithms that sample differently, and use this one when comparing against the definition in the literature.
    """
    total_angle, count = total_turning(path)

    # Every segment was degenerate, so there is nothing to report
    if count == 0:
        return 0.0

    # Divide by the maximum possible total so the result lands between 0 and 1
    return float(total_angle / (count * np.pi))


def curvature(path):
    """
    The average turning per cell driven, in radians. Lower is better.

    Dividing the total turning by the length of the route instead of by the number of points makes this measure depend on the shape of the route and not on how finely the planner happened to sample it. Sampling the same corner with six points or with one gives the same answer, which is what makes it fair to compare APF against RRT* and to compare a smoothed route against the route it came from.
    """
    total_angle, count = total_turning(path)

    # Nothing turned, or there was nothing to turn along
    if count == 0:
        return 0.0

    length = path_length(path)
    if length < 1e-9:
        return 0.0

    return float(total_angle / length)


def sharp_turns(path, threshold):
    """
    How many vertices turn by more than 'threshold' radians. This is the count of corners the smoothers would treat as sharp, so on a smoothed route it says how many corners the smoothing left behind.
    """
    # Fewer than three points means there is no interior vertex to turn at
    if path is None or len(path) < 3:
        return 0

    count = 0

    for i in range(1, len(path) - 1):
        before = np.asarray(path[i - 1], dtype=float)
        here = np.asarray(path[i], dtype=float)
        after = np.asarray(path[i + 1], dtype=float)

        incoming = here - before
        outgoing = after - here

        length_in = np.linalg.norm(incoming)
        length_out = np.linalg.norm(outgoing)

        # Repeated points have no direction to turn from or to
        if length_in < 1e-9 or length_out < 1e-9:
            continue

        cos_angle = np.clip(
            np.dot(incoming, outgoing) / (length_in * length_out), -1.0, 1.0
        )

        if np.arccos(cos_angle) > threshold:
            count += 1

    return count


def radius_violated(path, min_radius, spacing=1.0):
    """
    True when the route somewhere turns tighter than a vehicle with the given minimum turning radius can. The tightest turn is max_curvature, in radians per cell, and a radius of R cells allows at most 1 / R of it.
    """
    if min_radius <= 0:
        return False

    return bool(max_curvature(path, spacing) > 1.0 / min_radius)


def resample_evenly(path, spacing=1.0):
    """
    Return the path with its points respaced evenly by distance, at roughly one point every 'spacing' cells.

    The new points lie on the original polyline, so the shape of the route is unchanged and only the spacing of the points along it moves. Measuring a turn needs this: the planners space their points very differently, and a turn measured as radians per cell of segment is smaller when the segments happen to be longer. Respacing first puts every route on the same footing.
    """
    points = np.asarray(path, dtype=float)

    # Too short to respace, so hand it back as it is
    if len(points) < 3:
        return points

    # The running distance along the path at each original point
    hops = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(hops)])
    total = cumulative[-1]

    if total < 1e-9:
        return points

    # How many points the respaced path needs, keeping at least three so there is an interior vertex left to measure
    count = max(3, int(round(total / spacing)) + 1)

    # Interpolate each coordinate against the running distance
    targets = np.linspace(0.0, total, count)
    xs = np.interp(targets, cumulative, points[:, 0])
    ys = np.interp(targets, cumulative, points[:, 1])

    return np.column_stack([xs, ys])


def max_curvature(path, spacing=1.0):
    """
    The tightest turn anywhere on the route, in radians per cell. Lower is better. A value of k means the tightest part of the route needs a turning radius of 1 / k cells, so a vehicle that cannot turn tighter than radius R needs this to stay under 1 / R.

    This is the measure that shows whether smoothing did its job. Rounding a corner does not change how much the route turns in total, since that is fixed by the direction it arrives on and the direction it leaves on. What rounding changes is where the turning happens: a sharp corner puts the whole turn at a single point, and a rounded one spreads it along an arc.

    The path is respaced evenly before anything is measured. Without that the answer would depend on how finely each planner sampled its route rather than on the route, and a smoother that packs its points closely would be reported as turning more tightly than the sharp corner it replaced.

    The turn at a vertex of a polyline has no single value, so any measure of it has to pick a scale to look at, and the spacing is that scale. One cell is the default because that is the resolution of the map. Read the result as the tightest turn the route demands of a vehicle that follows it to within a cell. A route that zigzags inside a single cell is reported as turning sharply, which is the right answer for a vehicle that has to drive it.
    """
    points = resample_evenly(path, spacing)

    # Nothing with an interior vertex to turn at
    if len(points) < 3:
        return 0.0

    tightest = 0.0

    for i in range(1, len(points) - 1):
        # The segment vectors arriving at and leaving this vertex
        incoming = points[i] - points[i - 1]
        outgoing = points[i + 1] - points[i]

        length_in = np.linalg.norm(incoming)
        length_out = np.linalg.norm(outgoing)

        # Skip degenerate segments, which have no direction to turn from or to
        if length_in < 1e-9 or length_out < 1e-9:
            continue

        cos_angle = np.clip(
            np.dot(incoming, outgoing) / (length_in * length_out), -1.0, 1.0
        )
        angle = np.arccos(cos_angle)

        # The arc length the turn is spread over, which is half of each segment meeting at this vertex
        arc = 0.5 * (length_in + length_out)

        tightest = max(tightest, angle / arc)

    return float(tightest)


def clearance(grid, path):
    """
    How far the path stays away from obstacles, returned as (minimum, mean) in cells. Higher is safer.

    The distance is to the centre of the nearest real obstacle cell, a building, rubble or the moving obstacle. The margin the planners see is not an obstacle here: it is the vehicle's own half-width painted onto the map, and counting it would put every route at the same floor of one cell, since every route passes one cell off a margin somewhere. The distance field is read between the cells rather than at the nearest one, so a route is measured where it runs and not where it rounds to.
    """
    # Nothing to measure without a path
    if path is None or len(path) == 0:
        return 0.0, 0.0

    # The euclidean distance from every cell to the nearest real obstacle cell. The margin counts as free space for this
    real_obstacle = (grid != 0) & (grid != MARGIN_VALUE)
    dist_field = distance_transform_edt(~real_obstacle)

    # Grid dimensions for keeping the points inside the field
    h, w = grid.shape

    points = np.asarray(path, dtype=float)
    xs = np.clip(points[:, 0], 0, w - 1)
    ys = np.clip(points[:, 1], 0, h - 1)

    # Linear interpolation between the four cells around each point. The field is indexed (row, column), so y comes first
    values = map_coordinates(dist_field, [ys, xs], order=1, mode="nearest")

    return float(np.min(values)), float(np.mean(values))


def reached_goal(path, goal, tolerance):
    """
    Return True if the last point of the path is within the tolerance of the goal. The planners report their own success, and this is the independent check the runner uses so a planner cannot mark itself successful without arriving.
    """
    # Nothing to check without a path
    if path is None or len(path) == 0:
        return False

    # Convert both to numpy arrays of type float and compare the distance against the tolerance
    last = np.asarray(path[-1], dtype=float)
    goal = np.asarray(goal, dtype=float)

    return bool(np.linalg.norm(last - goal) <= tolerance)


def drive_time(path, speed, lateral_accel, spacing=1.0):
    """
    Seconds to drive the route when the vehicle slows for the turns. Lower is better.

    The constant speed model in mission_time cannot tell a zigzag from a curve of the same length. Here the route is respaced evenly, the turn at each vertex is read as a curvature the same way max_curvature reads it, and each segment is driven at the speed a vehicle can hold on that curvature without exceeding lateral_accel sideways: the square root of lateral_accel over the curvature, capped at the cruising speed. A straight route comes out the same as the constant speed model, so the two agree where there is nothing to disagree about.
    """
    if path is None or len(path) < 2 or speed <= 0:
        return 0.0

    points = resample_evenly(path, spacing)

    # The curvature at each vertex, zero at the two ends where there is no turn
    turning = np.zeros(len(points))

    for i in range(1, len(points) - 1):
        incoming = points[i] - points[i - 1]
        outgoing = points[i + 1] - points[i]

        length_in = np.linalg.norm(incoming)
        length_out = np.linalg.norm(outgoing)

        if length_in < 1e-9 or length_out < 1e-9:
            continue

        cos_angle = np.clip(
            np.dot(incoming, outgoing) / (length_in * length_out), -1.0, 1.0
        )
        turning[i] = np.arccos(cos_angle) / (0.5 * (length_in + length_out))

    total = 0.0

    for i in range(len(points) - 1):
        # A segment is driven at whatever its tighter end allows
        tightest = max(turning[i], turning[i + 1])

        if tightest > 1e-9:
            allowed = min(speed, np.sqrt(lateral_accel / tightest))
        else:
            allowed = speed

        total += np.linalg.norm(points[i + 1] - points[i]) / allowed

    return float(total)


def time_to_detection(path, junctions, pocket):
    """
    How many path points the vehicle spent inside the pocket before its first escape, or nan when that cannot be measured: no pocket on this map, no escape, an escape before the pocket was entered, or a route that never entered it.

    A point is inside the pocket when it lies in the (x_min, x_max, y_min, y_max) rectangle, inclusive. The count is in path points, which is APF steps taken; a step the hybrid refused because it would have hit a wall leaves no point and is not counted.
    """
    if pocket is None or not junctions or path is None:
        return float("nan")

    first_switch = int(junctions[0])
    x_min, x_max, y_min, y_max = pocket

    for index in range(min(first_switch, len(path))):
        x, y = path[index][0], path[index][1]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return first_switch - index

    return float("nan")


def mission_time(distance, comp_time, speed):
    """
    Total time from the call coming in to the vehicle arriving, in seconds: the time spent planning plus the time spent driving.

    Planning time on its own flatters a planner that thinks quickly and then drives a long way round, and path length on its own flatters one that finds a short route slowly. This is the number that decides whether anyone is reached in time, which is what the project is about.

    It assumes the vehicle drives the whole route at a constant speed. It has no acceleration, it does not slow for corners, and it does not stop. The comparison between planners is still fair because every planner is charged the same way, but the value is not a prediction of how long a real response would take.
    """
    # A speed of zero would make the drive infinitely long, which is not a useful answer
    if speed <= 0:
        return float("nan")

    return float(comp_time + distance / speed)


def score_run(grid, path, goal, config, reported_success):
    """
    Measure one finished run and return every metric as a dictionary.

    A run counts as successful only when the planner reported success, the last point is at the goal and the whole path is collision free. Checking all three stops a smoother or a fallback from quietly passing off a path that clips a building.
    """
    # The independent success check, which is the only definition of success used in the results
    success = (
        bool(reported_success)
        and reached_goal(path, goal, config.goal_tolerance)
        and validate_path(grid, path)
    )

    # A run that produced no usable path is scored as zero rather than left out, so every algorithm has the same number of rows
    if path is None or len(path) < 2:
        return {
            "success": success,
            "path_length": 0.0,
            "smoothness": 0.0,
            "curvature": 0.0,
            "max_curvature": 0.0,
            "clearance_min": 0.0,
            "clearance_mean": 0.0,
            "total_turning": 0.0,
            "sharp_turns": 0,
            "radius_violated": False,
            "drive_time_curved": 0.0,
        }

    clearance_min, clearance_mean = clearance(grid, path)
    turning, _ = total_turning(path)

    return {
        "success": success,
        "path_length": path_length(path),
        "smoothness": smoothness(path),
        "curvature": curvature(path),
        "max_curvature": max_curvature(path),
        "clearance_min": clearance_min,
        "clearance_mean": clearance_mean,
        "total_turning": turning,
        "sharp_turns": sharp_turns(path, config.turn_threshold),
        "radius_violated": radius_violated(path, config.dubins_radius),
        "drive_time_curved": drive_time(path, config.vehicle_speed, config.lateral_accel),
    }
