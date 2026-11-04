import unittest

from helpers import make_hand
from stillpoint.controller import Click, Controller, ControllerConfig, DrawStroke, ModeChanged, SetVolume, SwitchTab
from stillpoint.gestures import ModeMachine, pattern_to_mode
from stillpoint.legacy import LegacyModeSwitcher, fingers_to_mode, legacy_fingers_up

FPS = 30.0
MOUSE, VOLUME, TABS = (0, 1, 0, 0, 0), (1, 1, 0, 0, 0), (0, 1, 1, 0, 0)


class ModeMachineTests(unittest.TestCase):
    def test_switch_needs_continuous_dwell(self):
        m = ModeMachine(initial="mouse", dwell_s=0.2)
        events = [m.update(VOLUME, k / FPS) for k in range(7)]
        self.assertTrue(all(e is None for e in events[:6]))
        self.assertEqual(events[6].mode, "volume")

    def test_one_frame_flicker_never_switches(self):
        m = ModeMachine(initial="mouse", dwell_s=0.2)
        for k in range(300):
            states = VOLUME if k % 4 == 0 else MOUSE
            self.assertIsNone(m.update(states, k / FPS))
        self.assertEqual(m.mode, "mouse")

    def test_pinch_lock_freezes_the_mode(self):
        m = ModeMachine(initial="mouse", dwell_s=0.2)
        for k in range(30):
            self.assertIsNone(m.update(VOLUME, k / FPS, pinch_locked=True))
        self.assertEqual(m.mode, "mouse")

    def test_unrecognised_pattern_keeps_mode(self):
        self.assertIsNone(pattern_to_mode((1, 0, 1, 0, 1)))
        m = ModeMachine(initial="tabs")
        for k in range(30):
            m.update((1, 0, 1, 0, 1), k / FPS)
        self.assertEqual(m.mode, "tabs")


class LegacyTests(unittest.TestCase):
    def test_original_rules_on_upright_right_hand(self):
        for pose, mode in (("mouse", "mouse"), ("volume", "volume"), ("draw", "draw"), ("tabs", "tabs")):
            with self.subTest(pose=pose):
                self.assertEqual(fingers_to_mode(legacy_fingers_up(make_hand(pose))), mode)

    def test_original_thumb_rule_flips_for_a_left_hand(self):
        self.assertEqual(fingers_to_mode(legacy_fingers_up(make_hand("mouse", left=True))), "volume")

    def test_original_cooldown(self):
        sw = LegacyModeSwitcher(initial="mouse")
        self.assertEqual(sw.update([1, 1, 0, 0, 0], 0.0), "volume")
        self.assertEqual(sw.update([0, 1, 0, 0, 0], 0.3), "volume")  # inside 0.5 s cooldown
        self.assertEqual(sw.update([0, 1, 0, 0, 0], 0.6), "mouse")


class ControllerTests(unittest.TestCase):
    def _run(self, controller, hand, frames, t0=0.0):
        actions = []
        for k in range(frames):
            actions.extend(controller.step(hand, t0 + k / FPS))
        return actions

    def test_each_pose_reaches_its_mode(self):
        for pose in ("mouse", "volume", "draw", "tabs"):
            with self.subTest(pose=pose):
                c = Controller()
                actions = self._run(c, make_hand(pose), 10)
                self.assertEqual(c.mode, pose)
                self.assertIn(ModeChanged(pose), actions)

    def test_volume_follows_aperture(self):
        c = Controller()
        actions = self._run(c, make_hand("volume"), 40)
        levels = [a.level for a in actions if isinstance(a, SetVolume)]
        self.assertTrue(levels)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in levels))

    def test_draw_emits_strokes_and_tabs_swipe_switches(self):
        c = Controller()
        actions = self._run(c, make_hand("draw"), 10)
        self.assertTrue(any(isinstance(a, DrawStroke) for a in actions))
        c = Controller()
        self._run(c, make_hand("tabs", at=(500.0, 360.0)), 10)
        actions = self._run(c, make_hand("tabs", at=(700.0, 360.0)), 3, t0=1.0)
        self.assertIn(SwitchTab(1), actions)

    def test_mouse_pinch_clicks_once_and_stays_in_mouse_mode(self):
        c = Controller(ControllerConfig(variant="stillpoint"))
        self._run(c, make_hand("mouse", 0.8), 30)
        from helpers import pinch_hand

        actions = []
        apertures = [0.65, 0.5, 0.35, 0.2, 0.1, 0.1, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.8, 0.8]
        for k, a in enumerate(apertures):
            hand = pinch_hand(0.8, a, 0.3) if a < 0.8 else make_hand("mouse", 0.8)
            actions.extend(c.step(hand, 1.0 + k / FPS))
        self.assertEqual(sum(isinstance(a, Click) for a in actions), 1)
        self.assertEqual(c.mode, "mouse")
        self.assertFalse(any(isinstance(a, ModeChanged) for a in actions))

    def test_lost_hand_returns_no_actions(self):
        self.assertEqual(Controller().step(None, 0.0), [])


if __name__ == "__main__":
    unittest.main()
