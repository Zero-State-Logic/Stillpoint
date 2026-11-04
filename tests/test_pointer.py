import unittest

import numpy as np

from helpers import make_hand, pinch_hand
from stillpoint.pointer import PointerConfig, PointerPipeline, map_to_screen, screen_to_camera

FPS = 30.0
CFG = PointerConfig()


def _scripted_pinch(variant: str, slip: float = 0.4, rest: float = 0.8, size: float = 130.0):
    """Hold still 1 s, close in 6 frames, hold 4, reopen in 6, hold 0.5 s."""
    pipe = PointerPipeline(variant, CFG)
    at = (640.0, 360.0)
    closing = list(np.linspace(rest, 0.10, 7)[1:])
    apertures = [rest] * 30 + closing + [0.10] * 4 + closing[::-1] + [rest] * 15
    frames = []
    for k, a in enumerate(apertures):
        lm = make_hand("mouse", rest, at=at, size=size) if a >= rest else pinch_hand(rest, a, slip, at=at, size=size)
        frames.append(pipe.update(lm, k / FPS))
    pre_onset = np.array(frames[29].cursor)
    clicks = [c for f in frames for c in f.clicks]
    return pre_onset, clicks, frames


class MappingTests(unittest.TestCase):
    def test_active_region_maps_to_full_screen_and_clamps(self):
        np.testing.assert_allclose(map_to_screen((100, 100), CFG), (0, 0))
        np.testing.assert_allclose(map_to_screen((1180, 620), CFG), (1920, 1080))
        np.testing.assert_allclose(map_to_screen((5, 700), CFG), (0, 1080))

    def test_inverse_mapping_round_trips(self):
        p = np.array([733.0, 211.0])
        np.testing.assert_allclose(map_to_screen(screen_to_camera(p, CFG), CFG), p)


class BaselineTests(unittest.TestCase):
    def test_baseline_reproduces_original_smoothing_from_origin(self):
        pipe = PointerPipeline("baseline", CFG)
        hand = make_hand("mouse", 0.9, at=(640.0, 360.0))
        target = map_to_screen((640.0, 360.0), CFG)
        expected = np.zeros(2)
        for k in range(6):
            expected = expected + (target - expected) / 5.0
            np.testing.assert_allclose(pipe.update(hand, k / FPS).cursor, expected)

    def test_baseline_clicks_under_30px_then_sleeps_then_clicks_again(self):
        pipe = PointerPipeline("baseline", CFG)
        open_hand = make_hand("mouse", 0.9, at=(640.0, 360.0), size=130)
        closed = pinch_hand(0.9, 0.1, 0.0, at=(640.0, 360.0), size=130)  # tips about 12 px apart
        for k in range(30):
            pipe.update(open_hand, k / FPS)
        first = pipe.update(closed, 30 / FPS)
        self.assertEqual(len(first.clicks), 1)
        skipped = [pipe.update(closed, k / FPS) for k in range(31, 38)]  # inside the 0.25 s sleep
        self.assertTrue(all(f.skipped and not f.clicks for f in skipped))
        again = pipe.update(closed, 38 / FPS)  # sleep over, still pinched -> clicks again
        self.assertEqual(len(again.clicks), 1)

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            PointerPipeline("magic")


class RewindTests(unittest.TestCase):
    def test_rewind_returns_pre_onset_cursor_in_scripted_pinch(self):
        pre_onset, clicks, frames = _scripted_pinch("stillpoint")
        self.assertEqual(len(clicks), 1)
        self.assertLess(np.linalg.norm(np.array([clicks[0].x, clicks[0].y]) - pre_onset), 0.5)
        self.assertTrue(any(f.frozen and f.pinch == "closing" for f in frames))  # froze before the press

    def test_without_rewind_the_click_slips(self):
        pre_onset, clicks, _ = _scripted_pinch("oneeuro+hysteresis")
        self.assertEqual(len(clicks), 1)
        self.assertGreater(np.linalg.norm(np.array([clicks[0].x, clicks[0].y]) - pre_onset), 5.0)

    def test_cursor_resumes_after_release(self):
        _, _, frames = _scripted_pinch("stillpoint")
        self.assertFalse(frames[-1].frozen)
        self.assertEqual(frames[-1].pinch, "open")

    def test_one_click_per_pinch_for_hysteresis_variants(self):
        for variant in ("oneeuro+hysteresis", "stillpoint"):
            with self.subTest(variant=variant):
                self.assertEqual(len(_scripted_pinch(variant)[1]), 1)


class LostHandTests(unittest.TestCase):
    def test_missing_hand_holds_cursor_and_resets_after_timeout(self):
        pipe = PointerPipeline("stillpoint", CFG)
        hand = make_hand("mouse", 0.8)
        for k in range(10):
            last = pipe.update(hand, k / FPS)
        gap = pipe.update(None, 10 / FPS)
        self.assertTrue(gap.skipped)
        self.assertEqual(gap.cursor, last.cursor)
        pipe.update(None, 2.0)
        self.assertEqual(len(pipe.clicker.buffer), 0)


if __name__ == "__main__":
    unittest.main()
