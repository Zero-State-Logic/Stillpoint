import json
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401
from stillpoint.charts import write_charts
from stillpoint.evaluate import run_main
from stillpoint.fitts import Condition
from stillpoint.participants import SimParams
from stillpoint.simmodes import roll_sweep

SMOKE = SimParams(conditions=(Condition(300.0, 40.0), Condition(600.0, 80.0)), hover_s=3.0, track_modes=True)


class EvaluationSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_main(n=2, blocks=1, seed=1, workers=1, params=SMOKE)
        cls.result["meta"] = {"participants": 2, "blocks": 1, "conditions": [{}, {}], "targets_per_circle": 13}

    def test_full_pipeline_slips_less_than_baseline(self):
        v = self.result["variants"]
        self.assertLess(v["stillpoint"]["median_slip_px"], v["baseline"]["median_slip_px"])
        self.assertLess(v["stillpoint"]["error_rate"], v["baseline"]["error_rate"])

    def test_hysteresis_variants_never_double_click(self):
        for variant in ("oneeuro+hysteresis", "stillpoint"):
            self.assertEqual(self.result["variants"][variant]["duplicate_clicks_per_min"], 0.0)

    def test_result_is_json_and_has_paired_tests(self):
        text = json.dumps(self.result)
        self.assertIn("paired_stillpoint_vs_baseline", text)
        self.assertIn("mode_stability", self.result)

    def test_charts_are_written_and_parse(self):
        import xml.etree.ElementTree as ET

        self.result["roll_sweep"] = roll_sweep(1, angles=(-30, 0, 30), samples=3)
        with tempfile.TemporaryDirectory() as tmp:
            written = write_charts(self.result, Path(tmp))
            self.assertEqual({p.name for p in written}, {"results.svg", "modes.svg"})
            for path in written:
                ET.parse(path)


if __name__ == "__main__":
    unittest.main()
