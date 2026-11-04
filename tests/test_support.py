"""Tracking adapters, statistics and the chart writer."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import helpers  # noqa: F401
from stillpoint.stats import bootstrap_ci, paired_summary, sign_flip_p
from stillpoint.svgchart import bar_table, line_panels, nice_ticks
from stillpoint.tracking import build_observations, normalized_to_pixels, select_hand


def _fake_points(offset=0.0):
    return [SimpleNamespace(x=0.1 + offset + i * 0.01, y=0.2 + i * 0.01, z=-0.01 * i) for i in range(21)]


class TrackingTests(unittest.TestCase):
    def test_normalized_to_pixels_scales_z_like_x(self):
        px = normalized_to_pixels(_fake_points(), 1280, 720)
        self.assertEqual(px.shape, (21, 3))
        self.assertAlmostEqual(px[0, 0], 128.0)
        self.assertAlmostEqual(px[0, 1], 144.0)
        self.assertAlmostEqual(px[5, 2], -0.05 * 1280)

    def test_wrong_landmark_count_raises(self):
        with self.assertRaises(ValueError):
            normalized_to_pixels(_fake_points()[:20], 640, 480)

    def test_build_and_select_by_handedness(self):
        tasks_style = [SimpleNamespace(category_name="Left", score=0.7)]
        solutions_style = [SimpleNamespace(label="Right", score=0.9)]
        obs = build_observations([_fake_points(), _fake_points(0.3)], [tasks_style, solutions_style], 640, 480)
        self.assertEqual([o.handedness for o in obs], ["Left", "Right"])
        self.assertEqual(select_hand(obs).handedness, "Right")
        self.assertEqual(select_hand(obs, "left").handedness, "Left")
        self.assertIsNone(select_hand([], "any"))
        with self.assertRaises(ValueError):
            select_hand(obs, "both")


class StatsTests(unittest.TestCase):
    def test_sign_flip_exact_extremes(self):
        self.assertAlmostEqual(sign_flip_p([1.0, 2.0, 3.0, 4.0]), 2 / 16)
        self.assertAlmostEqual(sign_flip_p([1.0, -1.0]), 1.0)

    def test_bootstrap_ci_brackets_the_mean(self):
        data = np.random.default_rng(1).normal(5.0, 1.0, 40)
        lo, hi = bootstrap_ci(data)
        self.assertLess(lo, data.mean())
        self.assertGreater(hi, data.mean())

    def test_paired_summary(self):
        s = paired_summary([1.0, 2.0, 3.0], [2.0, 3.0, 5.0])
        self.assertAlmostEqual(s["mean_diff"], 4.0 / 3.0)
        self.assertEqual(s["n"], 3)
        with self.assertRaises(ValueError):
            paired_summary([1.0], [1.0, 2.0])


class ChartTests(unittest.TestCase):
    def test_nice_ticks_cover_max(self):
        self.assertEqual(nice_ticks(23.0), [0.0, 10.0, 20.0, 30.0])
        self.assertGreaterEqual(nice_ticks(0.37)[-1], 0.37)

    def test_charts_parse_as_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.svg", Path(tmp) / "b.svg"
            bar_table(a, "T", "S", [("one", "#ACA0F4"), ("two", "#4E2DC4")],
                      [{"title": "m", "hint": "lower", "values": [3.0, float("nan")], "fmt": str}])
            line_panels(b, "T", "S", [{"title": "p", "xlabel": "x", "x": [0, 1], "ys": {"one": [1, 2]}, "fmt": str}],
                        [("one", "#ACA0F4")])
            ET.parse(a)
            ET.parse(b)


if __name__ == "__main__":
    unittest.main()
