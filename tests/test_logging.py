import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers  # noqa: F401
from stillpoint import analyze
from stillpoint.fitts import Condition, Trial, summarize_condition
from stillpoint.fittstask import FittsTask, TrialLogger, read_rows, row_to_trial

CONDITIONS = [Condition(200.0, 40.0), Condition(400.0, 40.0)]


def _play(task: FittsTask, rng, t0: float = 0.0, mt: float = 0.8, spread: float = 4.0) -> float:
    """Click every target with Gaussian scatter; returns the end time."""
    t = t0
    while not task.finished:
        target = task.current_target()
        t += mt
        task.on_click(t, target.x + rng.normal(0, spread), target.y + rng.normal(0, spread))
    return t


class FittsTaskTests(unittest.TestCase):
    def test_start_click_is_not_logged_and_rows_follow(self):
        task = FittsTask(CONDITIONS, "P01", "stillpoint")
        first = task.current_target()
        self.assertIsNone(task.on_click(0.0, first.x, first.y))
        nxt = task.current_target()
        row = task.on_click(0.7, nxt.x, nxt.y)
        self.assertEqual(row["trial"], 1)
        self.assertEqual(row["hit"], 1)
        self.assertAlmostEqual(row["movement_time"], 0.7)

    def test_finishes_after_all_circles(self):
        task = FittsTask(CONDITIONS, "P01", "baseline")
        _play(task, np.random.default_rng(0))
        self.assertTrue(task.finished)
        self.assertEqual(len(task.rows), 2 * 13)
        self.assertIsNone(task.current_target())

    def test_skipped_target_is_a_miss_without_endpoint(self):
        task = FittsTask(CONDITIONS[:1], "P01", "baseline")
        first = task.current_target()
        task.on_click(0.0, first.x, first.y)
        row = task.on_click(1.0, float("nan"), float("nan"))
        self.assertEqual(row["hit"], 0)
        trial = row_to_trial({k: str(v) for k, v in row.items()})
        self.assertIsNone(trial.endpoint)
        self.assertFalse(trial.hit)


class CsvRoundTripTests(unittest.TestCase):
    def test_log_then_analyze_matches_direct_computation(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as tmp:
            tasks = []
            for participant, variant, mt in (("P01", "baseline", 1.0), ("P01", "stillpoint", 0.8),
                                             ("P02", "baseline", 1.1), ("P02", "stillpoint", 0.9)):
                path = Path(tmp) / f"{participant}_{variant}.csv"
                with TrialLogger(path) as logger:
                    task = FittsTask(CONDITIONS, participant, variant, logger=logger)
                    _play(task, rng, mt=mt)
                tasks.append(task)
            rows = read_rows(sorted(Path(tmp).glob("*.csv")))
            self.assertEqual(len(rows), 4 * 26)
            summary = analyze.analyze_rows(rows)
            # Recompute stillpoint throughput directly from the in-memory rows.
            direct = []
            for task in tasks:
                if task.variant != "stillpoint":
                    continue
                trials = [row_to_trial({k: str(v) for k, v in r.items()}) for r in task.rows]
                per_cond = {}
                for t in trials:
                    per_cond.setdefault(t.condition, []).append(t)
                direct.append(np.mean([summarize_condition(ts).throughput for ts in per_cond.values()]))
            self.assertAlmostEqual(summary["stillpoint"]["throughput_bps"], float(np.mean(direct)), places=9)
            self.assertGreater(summary["stillpoint"]["throughput_bps"], summary["baseline"]["throughput_bps"])
            self.assertEqual(summary["baseline"]["participants"], 2)

    def test_cli_expands_globs_and_writes_json(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp:
            with TrialLogger(Path(tmp) / "P09_oneeuro.csv") as logger:
                _play(FittsTask(CONDITIONS, "P09", "oneeuro", logger=logger), rng)
            out = Path(tmp) / "summary.json"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                code = analyze.main([str(Path(tmp) / "*.csv"), "--json", str(out)])
            self.assertEqual(code, 0)
            self.assertIn("oneeuro", printed.getvalue())
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(math.isfinite(data["oneeuro"]["throughput_bps"]))

    def test_cli_rejects_missing_and_foreign_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            foreign = Path(tmp) / "other.csv"
            foreign.write_text("a,b\n1,2\n", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(analyze.main([str(Path(tmp) / "nope.csv")]), 2)
                self.assertEqual(analyze.main([str(foreign)]), 2)

    def test_logger_rejects_incomplete_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            with TrialLogger(Path(tmp) / "x.csv") as logger:
                with self.assertRaises(ValueError):
                    logger.write({"participant": "P01"})


class TrialTypeTests(unittest.TestCase):
    def test_row_to_trial_restores_geometry(self):
        row = {"distance": "200", "width": "40", "origin_x": "0", "origin_y": "0", "target_x": "200",
               "target_y": "0", "start_x": "1", "start_y": "0", "click_x": "198", "click_y": "3",
               "movement_time": "0.9"}
        trial = row_to_trial(row)
        self.assertIsInstance(trial, Trial)
        self.assertTrue(trial.hit)
        self.assertAlmostEqual(trial.deviation, -2.0)


if __name__ == "__main__":
    unittest.main()
