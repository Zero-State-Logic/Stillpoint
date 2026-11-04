import unittest

import numpy as np

import helpers  # noqa: F401  (puts the repo on sys.path)
from stillpoint.filters import ExponentialSmoother, OneEuroFilter, OneEuroParams, smoothing_factor

FPS = 30.0


class OneEuroTests(unittest.TestCase):
    def test_constant_input_passes_through_unchanged(self):
        f = OneEuroFilter(OneEuroParams(min_cutoff=0.6, beta=0.012))
        out = [f([812.5, 401.25], k / FPS) for k in range(60)]
        for value in out:
            np.testing.assert_allclose(value, [812.5, 401.25], atol=1e-9)

    def test_jitter_variance_drops_when_still(self):
        rng = np.random.default_rng(3)
        noisy = 500.0 + rng.normal(0.0, 3.0, size=(600, 2))
        f = OneEuroFilter(OneEuroParams(min_cutoff=0.6, beta=0.012))
        out = np.array([f(x, k / FPS) for k, x in enumerate(noisy)])
        raw_var = noisy[100:].var(axis=0).mean()
        filt_var = out[100:].var(axis=0).mean()
        self.assertLess(filt_var, 0.15 * raw_var)

    def test_lag_is_small_at_speed(self):
        # 1000 px/s ramp: One Euro must lag far less than the fixed smoother.
        speed = 1000.0
        times = np.arange(0, 2.0, 1.0 / FPS)
        one_euro = OneEuroFilter(OneEuroParams(min_cutoff=0.6, beta=0.012))
        fixed = ExponentialSmoother(5.0, initial=(0.0,))
        lag_1e = lag_fixed = 0.0
        for t in times:
            truth = speed * t
            lag_1e = truth - float(one_euro([truth], t)[0])
            lag_fixed = truth - float(fixed([truth], t)[0])
        self.assertLess(lag_1e / speed, 0.060)  # under 60 ms behind
        self.assertLess(lag_1e, 0.3 * lag_fixed)

    def test_output_is_a_new_array(self):
        f = OneEuroFilter()
        a = f([1.0, 2.0], 0.0)
        a[0] = 99.0
        np.testing.assert_allclose(f.value, [1.0, 2.0])

    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            OneEuroParams(min_cutoff=0.0)
        with self.assertRaises(ValueError):
            OneEuroParams(beta=-1.0)
        with self.assertRaises(ValueError):
            smoothing_factor(1.0, 0.0)

    def test_non_increasing_time_holds_value(self):
        f = OneEuroFilter()
        f([0.0], 1.0)
        np.testing.assert_allclose(f([100.0], 1.0), [0.0])


class ExponentialSmootherTests(unittest.TestCase):
    def test_matches_original_update_rule(self):
        s = ExponentialSmoother(5.0, initial=(0.0, 0.0))
        x = np.zeros(2)
        for _ in range(7):
            x = x + (np.array([1000.0, 500.0]) - x) / 5.0
            np.testing.assert_allclose(s([1000.0, 500.0]), x)

    def test_rejects_smoothing_below_one(self):
        with self.assertRaises(ValueError):
            ExponentialSmoother(0.5)


if __name__ == "__main__":
    unittest.main()
