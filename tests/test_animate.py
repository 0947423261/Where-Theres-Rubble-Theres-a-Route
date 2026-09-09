"""
tests/test_animate.py

The renderer composites each frame over a background drawn once instead of redrawing the figure. That is only allowed because the frames come out identical to a full redraw, so that is the thing tested, along with the frame counts the two entry points promise.
"""

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from planning.maps.moving_obstacle import MovingObstacle
from planning.utils.config import Config
from planning.utils.animate import _render_frames, animate_dynamic_run, animate_run
from planning.utils.visualise import mark_ends, render_grid


def small_grid():
    grid = np.zeros((20, 20), dtype=np.uint8)
    grid[8:12, 5:15] = 1
    return grid


class TestCompositingMatchesAFullDraw(unittest.TestCase):
    def scene(self):
        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
        render_grid(ax, small_grid())
        mark_ends(ax, (2, 2), (17, 17))
        line, = ax.plot([], [], color="#2f6fb3", linewidth=2.2, zorder=3, label="route")
        ax.scatter([10], [3], c="#e9a13b", s=70, marker="*", zorder=5, label="escape")
        ax.legend(loc="upper left", fontsize=7)
        ax.set_xlim(-1, 20)
        ax.set_ylim(-1, 20)
        fig.tight_layout()
        return fig, ax, line

    def test_the_last_frame_equals_a_full_redraw(self):
        # The route runs through the escape star and under the legend, which is where compositing in the wrong order would show
        xs = np.linspace(2, 17, 30)
        ys = np.linspace(2, 17, 30)

        fig, ax, line = self.scene()
        moving = [line] + list(ax.collections) + [ax.get_legend()]
        composited = _render_frames(fig, ax, moving, len(xs), lambda f: line.set_data(xs[: f + 1], ys[: f + 1]))[-1]
        plt.close(fig)

        fig, ax, line = self.scene()
        line.set_data(xs, ys)
        fig.canvas.draw()
        full = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)

        np.testing.assert_array_equal(composited, full)

    def test_a_tie_in_depth_is_broken_the_way_the_full_draw_breaks_it(self):
        # The trail at the same depth as the start marker, added after it, and driven straight over it. The full draw paints the marker first and the trail over it; the composite has to do the same
        fig, ax, line = self.scene()
        line.set_zorder(4)
        moving = [line] + list(ax.collections) + [ax.get_legend()]
        composited = _render_frames(fig, ax, moving, 2, lambda f: line.set_data([0.0, 4.0], [2.0, 2.0]))[-1]
        plt.close(fig)

        fig, ax, line = self.scene()
        line.set_zorder(4)
        line.set_data([0.0, 4.0], [2.0, 2.0])
        fig.canvas.draw()
        full = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)

        np.testing.assert_array_equal(composited, full)

    def test_the_first_frame_equals_a_full_redraw(self):
        fig, ax, line = self.scene()
        moving = [line] + list(ax.collections) + [ax.get_legend()]
        composited = _render_frames(fig, ax, moving, 3, lambda f: line.set_data([2.0], [2.0]))[0]
        plt.close(fig)

        fig, ax, line = self.scene()
        line.set_data([2.0], [2.0])
        fig.canvas.draw()
        full = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)

        np.testing.assert_array_equal(composited, full)


class TestTheEntryPoints(unittest.TestCase):
    def test_the_static_animation_has_the_frames_asked_for(self):
        result = {"path": [[2.0, 2.0], [6.0, 4.0], [12.0, 6.0], [17.0, 17.0]]}
        with tempfile.TemporaryDirectory() as folder:
            written = animate_run(small_grid(), (2, 2), (17, 17), result, Path(folder) / "run.gif", n_frames=8, fps=10)
            self.assertEqual(Image.open(written).n_frames, 8)

    def test_a_traced_hybrid_pauses_to_grow_its_escape_tree(self):
        # One junction at the second raw point with a three-event tree. The drive is eight frames and the tree gets one frame per event on top
        raw = [[2.0, 2.0], [6.0, 4.0], [12.0, 6.0], [17.0, 17.0]]
        events = [("add", (7.0, 5.0), (6.0, 4.0)), ("add", (9.0, 5.5), (7.0, 5.0)), ("rewire", (9.0, 5.5), (7.0, 5.0), (6.0, 4.0))]
        result = {"path": [[2.0, 2.0], [6.5, 4.5], [12.0, 6.0], [17.0, 17.0]], "raw_path": raw, "junctions": [1],
                  "trace": [{"junction": 1, "events": events}]}
        config = Config(out_dir="/tmp/test_outputs")
        with tempfile.TemporaryDirectory() as folder:
            written = animate_run(small_grid(), (2, 2), (17, 17), result, Path(folder) / "run.gif", n_frames=8, fps=10, config=config, field=True)
            self.assertEqual(Image.open(written).n_frames, 8 + 3)

    def test_a_traced_rrt_star_grows_before_it_drives(self):
        events = [("add", (5.0, 5.0), (2.0, 2.0)), ("add", (9.0, 9.0), (5.0, 5.0)), ("goal", (17.0, 17.0), (9.0, 9.0))]
        result = {"path": [[2.0, 2.0], [5.0, 5.0], [9.0, 9.0], [17.0, 17.0]], "trace": events}
        with tempfile.TemporaryDirectory() as folder:
            written = animate_run(small_grid(), (2, 2), (17, 17), result, Path(folder) / "run.gif", n_frames=8, fps=10)
            self.assertEqual(Image.open(written).n_frames, 8 + 3)

    def test_the_dynamic_animation_has_one_frame_per_kept_step(self):
        grid = small_grid()
        obstacle = MovingObstacle(grid.shape, seed=1, half_size=1, speed=0.0, start=(10.0, 15.0))
        path = [[2.0 + 0.5 * t, 2.0 + 0.5 * t] for t in range(30)]
        with tempfile.TemporaryDirectory() as folder:
            written = animate_dynamic_run(grid, (2, 2), (17, 17), obstacle, path, Path(folder) / "run.gif", collided_at=20, max_frames=10)
            # Thirty steps at a stride of three is ten frames, plus the final step
            self.assertEqual(Image.open(written).n_frames, 11)
