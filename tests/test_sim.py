import dataclasses
import unittest

import numpy as np

import helpers  # noqa: F401
from stillpoint.fitts import Condition
from stillpoint.participants import SimParams, make_participants, minimum_jerk
from stillpoint.sim import simulate_session

SMALL = SimParams(conditions=(Condition(300.0, 80.0),), hover_s=2.0, warmup_s=0.5)


class SimulatorTests(unittest.TestCase):
    def test_same_seed_same_session(self):
        p = make_participants(1, seed=5)[0]
        a = simulate_session(p, "stillpoint", SMALL, seed=5)
        b = simulate_session(p, "stillpoint", SMALL, seed=5)
        self.assertEqual(a, b)

    def test_different_seed_different_session(self):
        p = make_participants(1, seed=5)[0]
        a = simulate_session(p, "baseline", SMALL, seed=5)
        b = simulate_session(p, "baseline", SMALL, seed=6)
        self.assertNotEqual(a.slips, b.slips)

    def test_participants_are_deterministic(self):
        self.assertEqual(make_participants(4, 9), make_participants(4, 9))
        self.assertNotEqual(make_participants(4, 9), make_participants(4, 10))

    def test_session_shape(self):
        p = make_participants(1, seed=2)[0]
        r = simulate_session(p, "stillpoint", SMALL, seed=2)
        self.assertEqual(r.selections, SMALL.n_targets + 1)  # start click + 13 trials
        self.assertEqual(len(r.trials), SMALL.n_targets)
        self.assertGreater(r.duration_s, 10.0)

    def test_zero_slip_and_noise_free_baseline_has_tiny_slip(self):
        quiet = dataclasses.replace(SMALL, slip=0.0, jitter_px=0.0, drop_prob=0.0)
        p = dataclasses.replace(make_participants(1, seed=3)[0], slip_bias=0.0, tremor_px=0.0)
        r = simulate_session(p, "oneeuro+hysteresis", quiet, seed=3)
        self.assertLess(float(np.median(r.slips)), 3.0)

    def test_minimum_jerk_endpoints_and_midpoint(self):
        np.testing.assert_allclose(minimum_jerk((0, 0), (10, 20), 0.0), (0, 0))
        np.testing.assert_allclose(minimum_jerk((0, 0), (10, 20), 1.0), (10, 20))
        np.testing.assert_allclose(minimum_jerk((0, 0), (10, 20), 0.5), (5, 10))

    def test_invalid_params_rejected(self):
        with self.assertRaises(ValueError):
            SimParams(slip=1.5)
        with self.assertRaises(ValueError):
            make_participants(0, 1)


if __name__ == "__main__":
    unittest.main()
