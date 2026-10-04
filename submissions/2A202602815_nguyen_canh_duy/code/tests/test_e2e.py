"""Smoke test end-to-end trên CPU với dữ liệu giả: train T00 x3 seed -> F01 (alias) -> study (val) -> test một lần
-> chấm bằng eval.py -> results.xlsx. Chỉ kiểm tra ĐƯỜNG ỐNG chạy thông và các chốt chặn; số liệu không có ý nghĩa."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from helpers import make_fake_deepweeds
import experiments as X
import final as F
import results as R
import study as S
import train as TR
from evalref import ev


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._paths, cls._global = dict(X.PATHS), dict(X.GLOBAL)
        cls.tmp = tempfile.TemporaryDirectory()
        r = Path(cls.tmp.name)
        cls.root = r
        make_fake_deepweeds(r / "data", per_class=(12, 4, 4))
        X.set_paths(out_dir=r / "runs", pred_dir=r / "predictions", curves_dir=r / "curves",
                    labels_dir=r / "data" / "labels", images_dir=r / "data" / "images", num_workers=0)
        for sd in X.FINAL_SEEDS:
            TR.run(TR.Config(exp_id="T00", desc="base", backbone="resnet18", pretrained=False, img_size=64,
                             epochs=1, batch_size=16, seed=sd, amp=False, measure_latency=False,
                             expected_total=None, lr_backbone=1e-3, lr_head=3e-3,
                             images_dir=X.PATHS["images_dir"], labels_dir=X.PATHS["labels_dir"],
                             out_dir=X.PATHS["out_dir"], pred_dir=X.PATHS["pred_dir"],
                             curves_dir=X.PATHS["curves_dir"], num_workers=0))
        X.update_decisions(backbone_primary="resnet18", recipe_id="T00", recipe_overrides={},
                           noise={"std": 0.01}, threshold=0.003)
        X.run_final_training()

    @classmethod
    def tearDownClass(cls):
        X.PATHS.clear(); X.PATHS.update(cls._paths)
        X.GLOBAL.clear(); X.GLOBAL.update(cls._global)
        cls.tmp.cleanup()

    def test_1_final_is_alias_of_t00(self):
        s = X.load_summary("F01", 1)
        self.assertEqual(s["alias_of"], "T00")
        self.assertFalse((self.root / "runs" / "F01" / "seed1" / "test_started.json").exists())

    def test_2_study_writes_tables_and_decisions(self):
        out = S.run_study(sizes=(64, 80, 96, 112), device="cpu", skip_latency=True,
                          figures_dir=str(self.root / "figures"))
        t = out["inference_table"]
        for eid in ("I00", "I01", "I02a", "I07"):
            self.assertIn(eid, set(t["exp_id"]))
        d = X.load_decisions()
        self.assertIn("inference", d)
        self.assertIn("views", d["inference"])
        self.assertTrue((self.root / "runs" / "inference" / "inference_table.csv").exists())

    def test_3_test_once_guard_and_scoring(self):
        if "inference" not in X.load_decisions():
            S.run_study(sizes=(64, 80, 96, 112), device="cpu", skip_latency=True,
                        figures_dir=str(self.root / "figures"))
        temps = F.run_test_once(device="cpu")
        self.assertEqual(sorted(temps), list(X.FINAL_SEEDS))
        pred = self.root / "predictions"
        for tag in ("F01", "F01_uncal", "F01rt", "T00"):
            self.assertEqual(len(list(pred.glob(f"{tag}_seed*_test.csv"))), 3, tag)
        # lần thứ hai phải bị từ chối (chốt chặn test-once)
        with self.assertRaises(Exception):
            F.run_test_once(device="cpu")
        texts = F.score_and_grade(out_dir=str(self.root / "eval_out"))
        self.assertIn("grade", texts)
        F.error_analysis(figures_dir=str(self.root / "figures"), out_dir=str(self.root / "eval_out"))
        self.assertTrue((self.root / "figures" / "confusion_test.png").exists())

        x = self.root / "results.xlsx"
        R.build_results(str(x))
        with pd.ExcelFile(x) as xl:
            self.assertEqual(len(xl.sheet_names), 7)
        self.assertEqual(R.check_consistency(str(x)), [])
        g = ev.load_group(str(pred / "F01_seed*_test.csv"), str(self.root / "data" / "labels" / "test_subset0.csv"))
        self.assertEqual(len(g.preds), 3)


if __name__ == "__main__":
    unittest.main()
