"""
utils/animate.py

Turns a planner result into an animation of the vehicle driving the route, written out as a GIF or, where ffmpeg is installed, an MP4.

Nothing else in the project depends on this module. It only reads a grid, a path and an obstacle, so it can be extended or left out without touching the experiment.

Two details are worth knowing about:

Constant speed. The planners space their path points very differently, since APF takes thousands of tiny steps while RRT* takes a handful of long jumps. Playing the raw points back would make RRT* look like it teleports. The static animation resamples the route at even distances instead, so the vehicle drives at one speed whichever planner produced the route.

Steady heading. The direction between two consecutive frames jitters on a wiggly path, which makes the vehicle spin on the spot. The heading angles are unwrapped to remove the jumps where the angle crosses half a turn, then averaged over a few frames, so the vehicle steers instead of twitching.

The dynamic animation does not resample, because frame f has to show the vehicle at time step f next to the obstacle at time step f. Long runs are subsampled by taking every few steps, which keeps the two in step with each other and the file small.

Frames are composited rather than redrawn. Saving through matplotlib's animation writers redraws the whole figure for every frame, map and all, and that is where nearly all the time went. Here the parts of the scene that never move are drawn once and kept as a pixel buffer, and each frame restores that buffer and draws only the artists that move on top of it, in the same depth order the full draw would use. The frames come out pixel for pixel the same as a full redraw and take a fraction of the time. Everything at or above the trail's depth is treated as moving, the start and goal markers and the legend included, so nothing static is ever painted under something that should be in front of it.
"""

import subprocess

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import Affine2D
from PIL import Image

from ..algorithms.apf import APF
from .visualise import GOAL_COLOUR, JUNCTION_COLOUR, PATH_COLOUR, mark_ends, render_grid

# The colours of the things the static animation adds on top of the map: the tree RRT* grows, an edge it has just rewired, the field APF follows and the route before smoothing
TREE_COLOUR = "#9aa3ad"
REWIRE_COLOUR = "#e9a13b"
FIELD_COLOUR = "#b8bfc9"
RAW_COLOUR = "#7a3b8a"


def _resample_constant_speed(path, n_frames):
    """
    Return n_frames points spaced evenly by distance along the path.

    The cumulative distance at each original point is computed first, then linear interpolation reads off the position at each of n_frames evenly spaced distances between zero and the total length.
    """
    # The path as an array of shape (number of points, 2)
    points = np.asarray(path, dtype=float)

    # A path with one point has nowhere to drive, so every frame sits on it
    if len(points) < 2:
        return np.repeat(points, max(n_frames, 1), axis=0)

    # The length of each hop, then the running total along the path
    hops = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(hops)])
    total = cumulative[-1]

    # Every point sits on top of the last, so there is no distance to spread the frames along
    if total < 1e-9:
        return np.repeat(points[:1], n_frames, axis=0)

    # The distances along the path each frame should sit at
    targets = np.linspace(0.0, total, n_frames)

    # Interpolate each coordinate separately against the running total
    xs = np.interp(targets, cumulative, points[:, 0])
    ys = np.interp(targets, cumulative, points[:, 1])

    return np.column_stack([xs, ys])


def _headings(frames, smooth=7):
    """
    The angle in radians the vehicle should face at each frame.

    The raw heading is the direction from one frame to the next. Unwrapping removes the sudden jumps of a full turn that appear when the angle crosses the half turn boundary, and a short moving average takes out the jitter.
    """
    # The direction from each frame to the next
    steps = np.diff(frames, axis=0)
    theta = np.arctan2(steps[:, 1], steps[:, 0])

    # The final frame has no next frame, so it keeps the heading it arrived on
    theta = np.append(theta, theta[-1])

    # Remove the artificial jumps of a full turn
    theta = np.unwrap(theta)

    if smooth > 1:
        # A moving average over a few frames. The array is padded with its edge values first so the average at the ends is not dragged towards zero
        kernel = np.ones(smooth) / smooth
        padded = np.pad(theta, smooth, mode="edge")
        theta = np.convolve(padded, kernel, mode="same")[smooth:-smooth]

    return theta


def _build_ambulance(ax, scale):
    """
    Build the vehicle out of matplotlib patches, drawn in its own coordinates with the nose along the positive x axis and the centre at the origin. Each frame then rotates and moves the whole group into place with one transform.

    Seen from above it is a white body with a red outline, a dark windscreen at the front, an amber light bar on the nose and a red cross on the roof.
    """
    s = scale

    body = FancyBboxPatch(
        (-1.6 * s, -0.9 * s),
        3.2 * s,
        1.8 * s,
        boxstyle=f"round,pad=0,rounding_size={0.35 * s}",
        facecolor="white",
        edgecolor=GOAL_COLOUR,
        linewidth=1.6,
        zorder=6,
    )
    windscreen = Rectangle(
        (0.55 * s, -0.65 * s), 0.5 * s, 1.3 * s, facecolor="#3b4048", edgecolor="none", zorder=7
    )
    lightbar = Rectangle(
        (1.25 * s, -0.45 * s), 0.3 * s, 0.9 * s, facecolor=JUNCTION_COLOUR, edgecolor="none", zorder=7
    )
    cross_vertical = Rectangle(
        (-0.75 * s, -0.55 * s), 0.44 * s, 1.1 * s, facecolor=GOAL_COLOUR, edgecolor="none", zorder=7
    )
    cross_horizontal = Rectangle(
        (-1.08 * s, -0.22 * s), 1.1 * s, 0.44 * s, facecolor=GOAL_COLOUR, edgecolor="none", zorder=7
    )

    patches = [body, windscreen, lightbar, cross_vertical, cross_horizontal]

    for patch in patches:
        ax.add_patch(patch)

    return patches


def _vehicle_scale(grid):
    """The size of the vehicle in cells, chosen so it covers about one twenty-second of the map width whatever the grid size is."""
    return grid.shape[1] / 22.0 / 3.2


def _render_frames(fig, ax, moving, n_frames, update):
    """
    Return one RGBA array per frame, composited over a background drawn once.

    The moving artists are marked animated so the background draw leaves them out. For each frame the update function moves them, the background is restored and they are drawn on top in depth order, then the canvas buffer is copied out. Depth order is what makes the result match a full redraw: matplotlib draws an axis's artists sorted by zorder, ties in the order they were added, and the same sort over the same list is used here, so every pixel lands where the full draw would have put it.
    """
    for artist in moving:
        artist.set_animated(True)

    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)

    # Take the moving artists in the order the axis holds them, then sort by depth. sorted is stable, so two artists at the same depth are drawn in the order they were added, which is the tie rule the full draw uses
    order = sorted((artist for artist in ax.get_children() if artist in moving), key=lambda artist: artist.get_zorder())

    frames = []

    for f in range(n_frames):
        update(f)
        fig.canvas.restore_region(background)
        for artist in order:
            ax.draw_artist(artist)
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())

    return frames


def _write_frames(frames, save_path, fps):
    """
    Write RGBA frames out as a GIF, or as an MP4 through ffmpeg when the file name asks for one.

    The GIF is written the way matplotlib's Pillow writer writes it, one image per frame at the playback rate, looping. The MP4 is raw frames piped into ffmpeg, which is what matplotlib's ffmpeg writer does too, without the redraw per frame that comes with it.
    """
    save_path = str(save_path)
    height, width = frames[0].shape[:2]

    if save_path.lower().endswith(".mp4"):
        # yuv420p is what most players expect, and it needs even dimensions, so the frame is padded by a pixel where it is odd
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-vcodec", "h264", "-pix_fmt", "yuv420p", "-b:v", "1800k", save_path,
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            for frame in frames:
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to write {save_path}")
    else:
        # Matplotlib's Pillow writer hands an opaque frame to the GIF encoder as RGB, since that quantises to the GIF palette better than RGBA does. The same is done here so the file comes out as it did before
        images = []
        for frame in frames:
            image = Image.fromarray(frame, "RGBA")
            images.append(image if frame[:, :, 3].min() < 255 else image.convert("RGB"))
        images[0].save(save_path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)

    return save_path


def _draw_field(ax, grid, start, goal, config):
    """
    Draw the direction APF would drive in from a lattice of free cells, as small grey arrows under everything else.

    The arrows are the resultant of the pull to the goal and the push off the obstacles, read off a throwaway APF planner so they are exactly the field the planner follows. Only the direction is drawn, since the magnitude near a wall dwarfs everything else. Where the arrows turn to face each other is where APF stalls, which is the whole reason the hybrid exists.
    """
    planner = APF(grid, start, goal, config, None)
    dist_stack, gx_stack, gy_stack = planner._precompute_fields()

    # One arrow every few cells, on free cells only
    h, w = grid.shape
    spacing = max(2, w // 24)
    xs, ys, us, vs = [], [], [], []
    for y in range(spacing // 2, h, spacing):
        for x in range(spacing // 2, w, spacing):
            if grid[y, x] != 0:
                continue
            force = planner._apf_force(np.array([x, y], dtype=float), planner.goal, dist_stack, gx_stack, gy_stack)
            magnitude = np.hypot(force[0], force[1])
            if magnitude < 1e-9:
                continue
            xs.append(x)
            ys.append(y)
            us.append(force[0] / magnitude)
            vs.append(force[1] / magnitude)

    ax.quiver(
        xs, ys, us, vs, color=FIELD_COLOUR, angles="xy", scale_units="xy", scale=1.0 / (0.7 * spacing),
        width=0.004, headwidth=3.5, zorder=1.5,
    )


def _drive_schedule(path, raw_path, junctions, n_frames):
    """
    Return (frames, theta, switch_at): the vehicle's positions and headings for the drive, and for each junction the drive frame it falls on.

    The junction indices point into the route before smoothing, and the vehicle drives the smoothed one, so each junction is placed at the drive frame closest to where it happened.
    """
    frames = _resample_constant_speed(path, n_frames)
    theta = _headings(frames)

    switch_at = []
    for junction in junctions:
        point = np.asarray(raw_path[junction], dtype=float)
        switch_at.append(int(np.argmin(np.linalg.norm(frames - point, axis=1))))

    return frames, theta, switch_at


def _tree_segments(parent_of):
    """The edges of a tree as a list of ((x, y), (x, y)) pairs, one per node with a parent."""
    return [(node, parent) for node, parent in parent_of.items()]


def animate_run(grid, start, goal, result, save_path, title="", n_frames=150, fps=25, config=None, field=False):
    """
    Animate one static planner run and write it to save_path.

    grid       the 2D map
    start,goal the two ends as (x, y)
    result     the planner's result dictionary: path, and when present raw_path, junctions and trace
    n_frames   frames spent driving the route; growing a tree adds frames on top
    fps        playback speed
    config     the configuration, needed to draw the APF field
    field      draw the APF field under the map, for the planners that follow it

    What is shown, and when. The vehicle drives the finished route at constant speed. Where the hybrid switched to RRT*, the drive pauses and the escape's tree grows edge by edge, with an edge that has just been rewired lit in amber, and the mode readout in the corner says which planner is in charge. A standalone RRT* run grows its whole tree before the drive. A smoothed route is driven over a dashed copy of the route before smoothing, so the smoother's work is legible. Axis limits are the map's, never rescaled.
    """
    path = result.get("path")

    # There is nothing to animate when the planner returned no usable route
    if path is None or len(path) < 2:
        raise ValueError(
            "Cannot animate this run: the planner produced no usable path, which usually means it failed on this map."
        )

    raw_path = result.get("raw_path", path)
    junctions = list(result.get("junctions", []))

    # The trees to grow, each with the drive frame it starts at. A hybrid grows one per escape at its junction; a standalone RRT* grows its one tree before moving
    frames, theta, switch_at = _drive_schedule(path, raw_path, junctions, n_frames)
    trace = result.get("trace", [])
    if trace and isinstance(trace[0], tuple):
        escapes = [(0, trace)]
    else:
        escapes = [(switch_at[i], escape["events"]) for i, escape in enumerate(trace) if i < len(switch_at)]

    # Which planner this is, read off the result: a standalone RRT* traces a flat list of events, a hybrid reports junctions, and APF reports neither
    hybrid = "junctions" in result
    stalled = not result.get("success", True)

    # Half as many frames again for growing trees, shared between escapes by how many events each has, so a long escape grows for longer. Never more frames than events, since a frame that adds nothing is a frame the GIF writer folds into the one before
    grow_total = n_frames // 2 if escapes else 0
    total_events = sum(len(events) for _, events in escapes)
    grow_frames = [
        max(1, min(len(events), max(6, int(round(grow_total * len(events) / max(1, total_events))))))
        for _, events in escapes
    ]

    ### === The static part of the scene === ###
    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=100)
    render_grid(ax, grid)
    mark_ends(ax, start, goal)
    end_markers = list(ax.collections)

    if field and config is not None:
        _draw_field(ax, grid, start, goal, config)

    # The route before smoothing, when there is one, so the smoothed drive can be read against it
    raw = np.asarray(raw_path, dtype=float)
    smoothed = raw_path is not path and not (len(raw) == len(path) and np.allclose(raw, np.asarray(path, dtype=float)))
    if smoothed:
        ax.plot(raw[:, 0], raw[:, 1], color=RAW_COLOUR, linewidth=1.0, linestyle="--", zorder=2.55, label="before smoothing")

    # The points where the hybrid handed control to RRT*
    junction_marker = None
    if junctions:
        points = np.asarray([raw_path[j] for j in junctions], dtype=float)
        junction_marker = ax.scatter(points[:, 0], points[:, 1], c=JUNCTION_COLOUR, s=70, marker="*", zorder=5, label="RRT* escape")

    ax.set_title(title, fontsize=11)
    ax.set_xlim(-1, grid.shape[1])
    ax.set_ylim(-1, grid.shape[0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ### === END ===

    # Where a planner that never arrived came to rest, ringed so the stall is the thing the eye lands on
    stall_marker = None
    if stalled:
        stall_marker = ax.scatter([path[-1][0]], [path[-1][1]], s=380, facecolors="none", edgecolors=GOAL_COLOUR, linewidths=2.2, zorder=5, label="stalled here")

    # The things that change from frame to frame: the tree so far, the edges just rewired, the trail behind the vehicle, the mode readout and the vehicle itself. The tree only gets a legend entry when there is a tree to show
    tree = LineCollection([], colors=TREE_COLOUR, linewidths=0.6, zorder=2.6, label="RRT* tree" if escapes else None)
    ax.add_collection(tree)
    rewired = LineCollection([], colors=REWIRE_COLOUR, linewidths=1.4, zorder=2.7, label="rewired" if escapes else None)
    ax.add_collection(rewired)
    trail, = ax.plot([], [], color=PATH_COLOUR, linewidth=2.2, zorder=3, label="route")
    mode = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=8, zorder=5.5,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85))
    ambulance = _build_ambulance(ax, _vehicle_scale(grid))

    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout()

    ### === The frame list === ###
    # Each entry is ("grow", escape number, events so far, events in this chunk) or ("drive", frame index). A tree grows before the drive frame it belongs to and stays on screen afterwards
    timeline = []
    parent_of = {}
    for f in range(n_frames):
        for number, (at, events) in enumerate(escapes, start=1):
            if at != f:
                continue
            chunks = np.array_split(np.arange(len(events)), grow_frames[number - 1])
            for chunk in chunks:
                timeline.append(("grow", number, [events[i] for i in chunk], len(events)))
        timeline.append(("drive", f))
    ### === END ===

    def place_vehicle(f):
        move = Affine2D().rotate(theta[f]).translate(frames[f, 0], frames[f, 1]) + ax.transData
        for patch in ambulance:
            patch.set_transform(move)

    def update(index):
        """Show one entry of the timeline."""
        entry = timeline[index]

        if entry[0] == "grow":
            _, number, chunk, count = entry
            flashes = []
            for event in chunk:
                if event[0] == "add":
                    parent_of[event[1]] = event[2]
                elif event[0] == "rewire":
                    parent_of[event[1]] = event[3]
                    flashes.append((event[1], event[3]))
                else:
                    parent_of[event[1]] = event[2]
            tree.set_segments(_tree_segments(parent_of))
            rewired.set_segments(flashes)
            grown_so_far = len(parent_of)
            label = f"stuck: RRT* escape {number} growing, {grown_so_far} nodes" if hybrid else f"RRT* growing: {grown_so_far} of {count} nodes"
            mode.set_text(label)
            return

        _, f = entry
        rewired.set_segments([])
        trail.set_data(frames[: f + 1, 0], frames[: f + 1, 1])
        place_vehicle(f)

        # Who is driving: APF for APF and for the hybrids between escapes, the finished tree's route for RRT*. A planner that never arrived says so once it gets where it stopped
        if stalled and f == n_frames - 1:
            mode.set_text("APF stalled here: local minimum, no route" if (hybrid or field) else "stopped here: no route")
        elif hybrid or field:
            mode.set_text("APF driving")
        else:
            mode.set_text("driving the RRT* route")

    # The vehicle starts at the first frame's position so a growing tree is drawn with it in place
    place_vehicle(0)

    moving = [tree, rewired, trail, mode] + ambulance + end_markers + [ax.get_legend()]
    for marker in (junction_marker, stall_marker):
        if marker is not None:
            moving.append(marker)

    frames_rgba = _render_frames(fig, ax, moving, len(timeline), update)
    written = _write_frames(frames_rgba, save_path, fps)
    plt.close(fig)

    print(f"[animate] wrote {written} ({len(timeline)} frames at {fps} fps)")

    return written


def animate_dynamic_run(
    base_grid, start, goal, obstacle, path, save_path, title="", collided_at=None, fps=20, max_frames=220
):
    """
    Animate a run against the moving obstacle and write it to save_path.

    Frame f shows the vehicle at time step f and the obstacle where it was at time step f, so what is on screen is the encounter as it actually happened rather than a reconstruction.

    base_grid   the static map, without the obstacle stamped into it
    obstacle    the MovingObstacle the vehicle was driving against
    path        the route actually driven, one point per time step
    collided_at optional time step where the obstacle caught the vehicle, marked with a cross
    max_frames  long runs are subsampled down to this many frames to keep the file small
    """
    # There is nothing to animate without a driven route
    if path is None or len(path) < 2:
        raise ValueError("Cannot animate this run: no route was driven.")

    # The path as numpy arrays of type float
    path = [np.asarray(point, dtype=float) for point in path]

    # Take every stride-th time step so a long run still fits in a small file, while the vehicle and the obstacle stay on the same clock
    stride = max(1, len(path) // max_frames)
    steps = list(range(0, len(path), stride))

    # Always finish on the last step, so the run does not appear to stop early
    if steps[-1] != len(path) - 1:
        steps.append(len(path) - 1)

    ### === The static part of the scene === ###
    fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=100)
    render_grid(ax, base_grid)
    mark_ends(ax, start, goal)

    ax.set_title(title, fontsize=12)
    ax.set_xlim(-1, base_grid.shape[1])
    ax.set_ylim(-1, base_grid.shape[0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ### === END ===

    # The trail behind the vehicle
    trail, = ax.plot([], [], color=PATH_COLOUR, linewidth=2.2, zorder=4, label="driven route")

    # The obstacle, drawn as an amber square the same size as the cells it blocks
    half = obstacle.half
    obstacle_patch = Rectangle(
        (0, 0),
        2 * half + 1,
        2 * half + 1,
        facecolor=JUNCTION_COLOUR,
        edgecolor="#8a5e12",
        linewidth=1.2,
        zorder=6,
        label="moving obstacle",
    )
    ax.add_patch(obstacle_patch)

    # The vehicle, and the cross that marks where it was hit
    ambulance = _build_ambulance(ax, _vehicle_scale(base_grid))
    hit_marker = ax.scatter([], [], s=260, marker="x", c=GOAL_COLOUR, linewidths=3, zorder=10)

    # The simulation clock, so the obstacle's position can be read against the time step it belongs to
    clock = ax.text(0.98, 0.02, "", transform=ax.transAxes, fontsize=9, ha="right", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85))

    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    def heading(step):
        """The direction the vehicle is travelling at this time step, taken from the points on either side of it."""
        after = path[min(step + 1, len(path) - 1)]
        before = path[max(step - 1, 0)]
        direction = after - before

        # A vehicle that has not moved keeps facing along the x axis rather than snapping to an undefined angle
        if np.hypot(direction[0], direction[1]) < 1e-9:
            return 0.0

        return np.arctan2(direction[1], direction[0])

    def update(f):
        """Draw one frame: the trail so far, the obstacle at this time step and the vehicle on top of it."""
        step = steps[f]

        # The trail up to this time step
        trail.set_data(
            [point[0] for point in path[: step + 1]],
            [point[1] for point in path[: step + 1]],
        )

        # The obstacle, positioned by its bottom left corner since that is what a Rectangle patch is anchored by
        center_x, center_y = obstacle.position_at(step)
        obstacle_patch.set_xy((center_x - half - 0.5, center_y - half - 0.5))

        # The vehicle, rotated to its heading and moved onto its position
        move = (
            Affine2D().rotate(heading(step)).translate(path[step][0], path[step][1])
            + ax.transData
        )
        for patch in ambulance:
            patch.set_transform(move)

        # Once the collision has happened the cross stays on screen for the rest of the animation
        if collided_at is not None and step >= collided_at:
            hit_marker.set_offsets([path[min(collided_at, len(path) - 1)]])
            clock.set_text(f"t = {step}   hit at t = {collided_at}")
        else:
            hit_marker.set_offsets(np.empty((0, 2)))
            clock.set_text(f"t = {step}")

    # Everything at or above the trail's depth is drawn per frame. The start and goal markers and the legend are included so they stay in front of the trail, as they are in a full draw
    moving = [trail, obstacle_patch, hit_marker, clock] + ambulance + list(ax.collections) + [ax.get_legend()]

    frames_rgba = _render_frames(fig, ax, moving, len(steps), update)
    written = _write_frames(frames_rgba, save_path, fps)
    plt.close(fig)

    print(f"[animate] wrote {written} ({len(steps)} frames at {fps} fps)")

    return written
