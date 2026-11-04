import unittest

import numpy as np

from helpers import make_hand
from stillpoint import handmodel as hm
from stillpoint.hand import (
    FingerStateTracker,
    FingerThresholds,
    as_landmarks,
    finger_curls,
    finger_states,
    normalized_aperture,
    palm_scale,
    thumb_spread,
)

EXPECTED = {
    "mouse": (0, 1, 0, 0, 0),
    "volume": (1, 1, 0, 0, 0),
    "draw": (1, 1, 1, 1, 1),
    "tabs": (0, 1, 1, 0, 0),
}


class GeometryTests(unittest.TestCase):
    def test_palm_scale_grows_with_hand_size(self):
        small = palm_scale(make_hand(size=80))
        big = palm_scale(make_hand(size=160))
        self.assertAlmostEqual(big / small, 2.0, places=6)

    def test_aperture_is_scale_invariant(self):
        near = normalized_aperture(make_hand(aperture=0.7, size=170))
        far = normalized_aperture(make_hand(aperture=0.7, size=85))
        self.assertAlmostEqual(near, far, places=9)

    def test_bad_shapes_and_nan_are_rejected(self):
        with self.assertRaises(ValueError):
            as_landmarks(np.zeros((20, 3)))
        bad = make_hand()
        bad[3, 0] = np.nan
        with self.assertRaises(ValueError):
            as_landmarks(bad)
        with self.assertRaises(ValueError):
            palm_scale(np.zeros((21, 3)))

    def test_curls_separate_straight_from_folded(self):
        curls = finger_curls(make_hand("mouse"))
        self.assertLess(curls[0], 30.0)
        self.assertTrue(np.all(curls[1:] > 120.0))


class FingerStateTests(unittest.TestCase):
    def test_every_pose_reads_correctly_upright(self):
        for pose, expected in EXPECTED.items():
            with self.subTest(pose=pose):
                self.assertEqual(finger_states(make_hand(pose)), expected)

    def test_states_invariant_to_rotating_the_landmarks(self):
        rng = np.random.default_rng(11)
        for pose, expected in EXPECTED.items():
            for _ in range(60):
                roll, pitch, yaw = rng.uniform(-180.0, 180.0, 3)
                hand = make_hand(pose, aperture=rng.uniform(0.6, 1.0), roll=roll, pitch=pitch, yaw=yaw,
                                 size=rng.uniform(70, 200))
                with self.subTest(pose=pose, roll=roll, pitch=pitch, yaw=yaw):
                    self.assertEqual(finger_states(hand), expected)

    def test_states_identical_for_left_and_right_hands(self):
        for pose, expected in EXPECTED.items():
            for roll in (-60.0, -20.0, 0.0, 35.0, 90.0):
                with self.subTest(pose=pose, roll=roll):
                    self.assertEqual(finger_states(make_hand(pose, roll=roll, left=True)), expected)

    def test_in_plane_rotation_of_pixels_keeps_thumb_spread(self):
        hand = make_hand("volume")
        angle = np.radians(73.0)
        rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        turned = (hand - hand[0]) @ rot.T + hand[0]
        self.assertAlmostEqual(thumb_spread(hand), thumb_spread(turned), places=9)

    def test_tracker_hysteresis_ignores_small_wobble(self):
        th = FingerThresholds()
        tracker = FingerStateTracker(th)
        base = make_hand("mouse")
        tracker.update(base)
        rng = np.random.default_rng(5)
        for _ in range(300):
            self.assertEqual(tracker.update(base + rng.normal(0, 1.5, base.shape)), EXPECTED["mouse"])

    def test_empty_bands_rejected(self):
        with self.assertRaises(ValueError):
            FingerThresholds(curl_extended_below=100.0, curl_folded_above=90.0)


class HandModelTests(unittest.TestCase):
    def test_pinch_moves_index_tip_by_slip_share(self):
        base = hm.pose_landmarks("mouse", 0.8)
        closed = hm.pinch_landmarks(base, 0.8, 0.1, slip=0.25)
        moved = np.linalg.norm(closed[hm.INDEX_TIP] - base[hm.INDEX_TIP])
        self.assertAlmostEqual(moved, 0.25 * 0.7 * hm.PALM_LOCAL, places=9)
        gap = np.linalg.norm(closed[hm.THUMB_TIP] - closed[hm.INDEX_TIP])
        self.assertAlmostEqual(gap, 0.1 * hm.PALM_LOCAL, places=9)

    def test_unknown_pose_and_bad_slip_raise(self):
        with self.assertRaises(KeyError):
            hm.pose_landmarks("salute")
        with self.assertRaises(ValueError):
            hm.pinch_landmarks(hm.pose_landmarks("mouse"), 0.8, 0.2, slip=1.5)


if __name__ == "__main__":
    unittest.main()
