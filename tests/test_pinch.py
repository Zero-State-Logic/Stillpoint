import unittest

import numpy as np

import helpers  # noqa: F401
from stillpoint.pinch import (
    ApertureVelocity,
    CursorSample,
    HysteresisPinch,
    PinchParams,
    RewindBuffer,
    SingleThresholdPinch,
)


def _transitions(detector, signal) -> int:
    return sum(1 for a in signal if detector.update(a) is not None)


class HysteresisTests(unittest.TestCase):
    def test_no_chatter_on_noisy_aperture_near_threshold(self):
        rng = np.random.default_rng(21)
        params = PinchParams(press_below=0.22, release_above=0.32)
        # Sits right on the press threshold with tracker-like noise (SD 0.02).
        signal = 0.22 + rng.normal(0.0, 0.02, 3000)
        hysteresis = _transitions(HysteresisPinch(params), signal)
        single = _transitions(SingleThresholdPinch(0.22), signal)
        self.assertLessEqual(hysteresis, 1)  # at most one press, never a release
        self.assertGreater(single, 500)  # the naive detector chatters

    def test_one_press_and_one_release_per_pinch(self):
        pinch = HysteresisPinch()
        closing = np.linspace(0.8, 0.1, 10)
        opening = closing[::-1]
        events = [pinch.update(a) for a in np.concatenate([closing, opening])]
        self.assertEqual([e for e in events if e], ["press", "release"])

    def test_invalid_thresholds_rejected(self):
        with self.assertRaises(ValueError):
            PinchParams(press_below=0.4, release_above=0.3)


class VelocityTests(unittest.TestCase):
    def test_tracks_closing_speed(self):
        vel = ApertureVelocity(alpha=1.0)
        vel.update(0.0, 0.8)
        self.assertAlmostEqual(vel.update(0.1, 0.6), -2.0)


class RewindBufferTests(unittest.TestCase):
    def test_closing_start_returns_last_sample_before_the_fall(self):
        buf = RewindBuffer(horizon_s=1.0)
        t = 0.0
        for _ in range(10):  # still, open hand
            buf.push(CursorSample(t, 500.0, 400.0, 0.80))
            t += 1 / 30
        for k, a in enumerate((0.70, 0.50, 0.30, 0.18)):  # closing, cursor dragged
            buf.push(CursorSample(t, 500.0 + 8 * (k + 1), 400.0 + 6 * (k + 1), a))
            t += 1 / 30
        start = buf.closing_start(still_speed=0.6, max_lookback_s=0.4)
        self.assertEqual((start.x, start.y), (500.0, 400.0))
        self.assertAlmostEqual(start.aperture, 0.80)

    def test_lookback_is_capped(self):
        buf = RewindBuffer(horizon_s=5.0)
        for k in range(60):  # slow, endless closing
            buf.push(CursorSample(k / 30, float(k), 0.0, 1.0 - 0.015 * k))
        start = buf.closing_start(still_speed=0.1, max_lookback_s=0.4)
        self.assertGreaterEqual(start.t, 59 / 30 - 0.4 - 1e-9)

    def test_horizon_drops_old_samples_and_empty_returns_none(self):
        buf = RewindBuffer(horizon_s=0.5)
        self.assertIsNone(buf.closing_start(0.6, 0.4))
        for k in range(60):
            buf.push(CursorSample(k / 30, 0.0, 0.0, 0.8))
        self.assertLessEqual(len(buf), 17)


if __name__ == "__main__":
    unittest.main()
