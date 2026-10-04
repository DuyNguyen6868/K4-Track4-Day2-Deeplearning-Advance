"""Test logic quyết định (experiments.py) và dựng results.xlsx (results.py) trên dữ liệu tổng hợp, không cần train."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import make_fake_deepweeds  # noqa: F401  (đồng thời thêm code/ vào sys.path)

import experiments as X
import results as R
from evalref import ev


def _summary(exp_id, f1, seed=0, backbone="resnet50", per_class=None):
    pc = per_class or [f1] * 9
    return {"exp_id": exp_id, "seed": seed, "backbone": backbone, "weight_tag": "resnet50.a1_in1k",
            "img_size": 224, "epochs": 12, "best_epoch": 9, "val_macro_f1": f1, "val_top1": f1 + 0.02,
            "val_balanced_acc": f1, "val_ece": 0.03, "val_f1_per_class": pc, "val_recall_per_class": pc,
            "train_time_per_epoch_s": 30.0, "params_M": 25.6, "gmacs": 4.1, "latency_b1_ms": 6.0,
            "latency_b1_p95_ms": 7.0, "gpu": "fake", "config": {}}


class PathsMixin:
    def setUp(self):
        self._saved = dict(X.PATHS)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        X.set_paths(out_dir=self.root / "runs", pred_dir=self.root / "predictions",
                    curves_dir=self.root / "curves", labels_dir=self.root / "data" / "labels",
                    images_dir=self.root / "data" / "images")

    def tearDown(self):
        X.PATHS.clear()
        X.PATHS.update(self._saved)
        self.tmp.cleanup()

    def put(self, exp_id, f1, seed=0, **kw):
        d = self.root / "runs" / exp_id / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(_summary(exp_id, f1, seed, **kw)), encoding="utf-8")


class TestDecisions(PathsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.put("T00", 0.900)
        X.update_decisions(noise={"std": 0.002, "f1": [0.9, 0.902, 0.898]})

    def test_threshold_uses_two_std_with_floor(self):
        self.put("T03", 0.9035)   # +0.0035 >= max(0.004, 0.003)? không: ngưỡng 0.004
        self.put("T07", 0.9045)   # +0.0045 thắng
        d = X.decide_winners()
        self.assertAlmostEqual(d["threshold"], 0.004)
        self.assertEqual(d["winners"], ["T07"])

    def test_floor_when_noise_tiny_or_nan(self):
        X.update_decisions(noise={"std": float("nan")})
        self.put("T03", 0.9031)
        d = X.decide_winners()
        self.assertAlmostEqual(d["threshold"], X.MIN_DELTA)
        self.assertEqual(d["winners"], ["T03"])

    def test_nonadoptable_never_wins(self):
        self.put("T13", 0.95)
        d = X.decide_winners()
        self.assertEqual(d["winners"], [])
        self.assertFalse(d["winners_table"]["T13"]["winner"])

    def test_combo_one_per_slot_and_conflict(self):
        for e, f in (("T03", 0.91), ("T04", 0.93), ("T05", 0.92), ("T09", 0.915), ("T10", 0.925)):
            self.put(e, f)
        d = X.decide_winners()
        ov, chosen = X.combo_overrides(d["winners"], d["winners_table"])
        self.assertIn("T04", chosen)          # aug: T04 hơn T03
        self.assertNotIn("T03", chosen)
        self.assertIn("T10", chosen)          # xung đột T09/T10: giữ Δ lớn hơn
        self.assertNotIn("T09", chosen)
        self.assertEqual(ov["aug"], "trivial")
        self.assertEqual(ov["sampler"], "balanced")

    def test_recipe_falls_back_to_t00_without_winner(self):
        self.put("T03", 0.901)
        X.decide_winners()
        d = X.decide_recipe()
        self.assertEqual(d["recipe_id"], "T00")
        self.assertEqual(d["recipe_overrides"], {})

    def test_recipe_prefers_simpler_when_close(self):
        for e, f in (("T03", 0.920), ("T07", 0.921)):
            self.put(e, f)
        X.decide_winners()
        self.put("T20", 0.9215)
        X.update_decisions(combo={"members": ["T03", "T07"], "overrides": {"aug": "color", "loss": "ls"}})
        d = X.decide_recipe()
        self.assertIn(d["recipe_id"], ("T03", "T07"))   # đơn giản hơn T20 khi cách nhau < ngưỡng
        self.assertNotEqual(d["recipe_id"], "T20")


def _fake_preds(path: Path, files: list[str], y: np.ndarray, acc: float, seed: int):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(9), size=len(y)) * 0.3
    right = rng.random(len(y)) < acc
    tgt = np.where(right, y, (y + 1) % 9)
    p[np.arange(len(y)), tgt] += 0.7
    p /= p.sum(1, keepdims=True)
    ev.save_predictions(path, files, y, p)


class TestResults(PathsMixin, unittest.TestCase):
    def build(self):
        make_fake_deepweeds(self.root / "data", per_class=(6, 3, 5))
        test = pd.read_csv(self.root / "data" / "labels" / "test_subset0.csv")
        files, y = test["Filename"].tolist(), test["Label"].to_numpy()
        pred = self.root / "predictions"
        for sd in (0, 1, 2):
            for tag, acc in (("F01", 0.9), ("T00", 0.8), ("F01_uncal", 0.9), ("F01rt", 0.85)):
                _fake_preds(pred / f"{tag}_seed{sd}_test.csv", files, y, acc, sd + sum(map(ord, tag)) % 7)
            _fake_preds(pred / f"F01_seed{sd}_val.csv", files, y, 0.9, 50 + sd)
        for e, f in (("B01", 0.90), ("B02", 0.91)):
            self.put(e, f, backbone=X.BACKBONE_RUNS[e])
        for sd, f in ((0, 0.900), (1, 0.902), (2, 0.898)):
            self.put("T00", f, seed=sd)
        self.put("T03", 0.915, per_class=[0.9] * 7 + [0.7, 0.6])
        X.update_decisions(backbone_primary="resnet50", noise={"std": 0.002})
        X.decide_winners()
        for tag in ("B01", "B02", "T00", "T03", "F01"):
            (self.root / "curves").mkdir(exist_ok=True)
            (self.root / "curves" / f"{tag}_x.png").write_bytes(b"png")
        inf = self.root / "runs" / "inference"
        inf.mkdir(parents=True)
        pd.DataFrame([{"exp_id": "I00", "method": "1 view", "model": "F01 seed0", "K": 1, "macro_f1_val": 0.9,
                       "top1_val": 0.92, "ece_val": 0.04, "nll_val": 0.3, "p50_ms": 6.0, "p95_ms": 7.0,
                       "p99_ms": 8.0, "throughput_b1_img_s": 166.0, "cost_vs_I00": 1.0, "note": ""},
                      {"exp_id": "I02a", "method": "hflip", "model": "F01 seed0", "K": 2, "macro_f1_val": 0.91,
                       "top1_val": 0.93, "ece_val": 0.03, "nll_val": 0.28, "p50_ms": 12.0, "p95_ms": 14.0,
                       "p99_ms": 15.0, "throughput_b1_img_s": 83.0, "cost_vs_I00": 2.0, "note": ""}]
                     ).to_csv(inf / "inference_table.csv", index=False)
        pd.DataFrame([{"config": "F01 fp32", "gpu": "fake", "dtype": "fp32", "batch": 1, "img_size": 224,
                       "fused_bn": False, "p50": 6.0, "p95": 7.0, "p99": 8.0, "images_per_s": 166.0,
                       "warmup": 10, "n": 50, "torch": "x"}]).to_csv(inf / "latency_table.csv", index=False)
        return self.root / "results.xlsx"

    def test_build_has_seven_sheets_and_consistent(self):
        x = self.build()
        R.build_results(str(x))
        with pd.ExcelFile(x) as xl:
            sheets = list(xl.sheet_names)
        self.assertEqual(sheets, ["Backbones", "Training", "Inference", "Final", "PerClass", "Latency", "Summary"])
        fin = pd.read_excel(x, sheet_name="Final")
        self.assertTrue(fin["seed"].astype(str).str.startswith("mean").sum() >= 2)
        self.assertEqual(R.check_consistency(str(x)), [])

    def test_final_numbers_match_eval(self):
        x = self.build()
        R.build_results(str(x))
        fin = pd.read_excel(x, sheet_name="Final")
        g = ev.load_group(str(self.root / "predictions" / "F01_seed*_test.csv"),
                          str(self.root / "data" / "labels" / "test_subset0.csv"))
        row = fin[(fin["exp_id"] == "F01") & (fin["seed"].astype(str) == "0")].iloc[0]
        self.assertAlmostEqual(float(row["macro-F1 test"]), g.metrics[0]["macro_f1"], places=9)

    def test_consistency_detects_missing_curve_and_seed(self):
        x = self.build()
        R.build_results(str(x))
        (self.root / "curves" / "B02_x.png").unlink()
        (self.root / "predictions" / "T00_seed2_test.csv").unlink()
        probs = " ".join(R.check_consistency(str(x)))
        self.assertIn("B02", probs)
        self.assertIn("T00", probs)

    def test_partial_results_do_not_crash(self):
        self.put("B01", 0.9, backbone="resnet50")
        x = self.root / "partial.xlsx"
        R.build_results(str(x))
        with pd.ExcelFile(x) as xl:
            self.assertEqual(len(xl.sheet_names), 7)
        R.write_report_tables(str(self.root / "t.md"))

    def test_report_tables_written(self):
        x = self.build()
        R.build_results(str(x))
        p = R.write_report_tables(str(self.root / "t.md"))
        txt = p.read_text(encoding="utf-8")
        self.assertIn("Chung kết", txt)
        self.assertIn("F01", txt)


if __name__ == "__main__":
    unittest.main()
