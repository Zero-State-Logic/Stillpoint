import math
import unittest

import numpy as np

import helpers  # noqa: F401
from stillpoint.fitts import Condition, Trial, circle_targets, mean_of_means, selection_order, summarize_condition

COND = Condition(100.0, 20.0)


def _trial(endpoint, mt, start=(0.0, 0.0)):
    return Trial(COND, (0.0, 0.0), (100.0, 0.0), start, endpoint, mt)


class MetricsTests(unittest.TestCase):
    def test_hand_computed_example(self):
        # dx = -2, +2, +1, -1 -> SD (n-1) = sqrt(10/3); De = 100; MT mean 0.55 s.
        trials = [
            _trial((98.0, 0.0), 0.50),
            _trial((102.0, 3.0), 0.60),
            _trial((101.0, -2.0), 0.55),
            _trial((99.0, 1.0), 0.55),
        ]
        s = summarize_condition(trials)
        we = 4.133 * math.sqrt(10.0 / 3.0)
        ide = math.log2(100.0 / we + 1.0)
        self.assertAlmostEqual(s.we, we, places=9)
        self.assertAlmostEqual(s.de, 100.0, places=9)
        self.assertAlmostEqual(s.ide, ide, places=9)
        self.assertAlmostEqual(s.movement_time, 0.55, places=12)
        self.assertAlmostEqual(s.throughput, ide / 0.55, places=9)
        self.assertAlmostEqual(s.throughput, 6.9692, places=3)
        self.assertEqual(s.error_rate, 0.0)

    def test_errors_and_missing_clicks(self):
        trials = [_trial((100.0, 15.0), 0.5), _trial(None, None), _trial((100.0, 0.0), 0.5), _trial((99.0, 0.0), 0.6)]
        s = summarize_condition(trials)
        self.assertAlmostEqual(s.error_rate, 0.5)  # one outside, one with no click
        self.assertEqual(s.n_used, 3)

    def test_effective_distance_uses_the_actual_start(self):
        t = _trial((100.0, 0.0), 0.5, start=(-10.0, 4.0))
        self.assertAlmostEqual(t.effective_distance, 110.0)

    def test_mean_of_means(self):
        a = summarize_condition([_trial((98.0, 0), 0.5), _trial((102.0, 0), 0.5)])
        b = summarize_condition([_trial((99.0, 0), 1.0), _trial((101.0, 0), 1.0)])
        self.assertAlmostEqual(mean_of_means({"p1": [a], "p2": [b]}), (a.throughput + b.throughput) / 2)

    def test_mixed_conditions_rejected(self):
        other = Trial(Condition(50.0, 10.0), (0, 0), (50, 0), None, (50, 0), 0.4)
        with self.assertRaises(ValueError):
            summarize_condition([_trial((100.0, 0), 0.5), other])


class LayoutTests(unittest.TestCase):
    def test_circle_has_requested_diameter(self):
        pts = circle_targets(13, 600.0, center=(0.0, 0.0))
        np.testing.assert_allclose(np.linalg.norm(pts, axis=1), 300.0)

    def test_order_visits_every_target_and_alternates(self):
        order = selection_order(13)
        self.assertEqual(sorted(order[:-1]), list(range(13)))
        self.assertEqual(order[0], order[-1])
        pts = circle_targets(13, 600.0, center=(0.0, 0.0))
        hops = [np.linalg.norm(pts[a] - pts[b]) for a, b in zip(order, order[1:])]
        self.assertTrue(all(h > 590.0 for h in hops))

    def test_even_target_count_rejected(self):
        with self.assertRaises(ValueError):
            circle_targets(12, 300.0)


if __name__ == "__main__":
    unittest.main()
