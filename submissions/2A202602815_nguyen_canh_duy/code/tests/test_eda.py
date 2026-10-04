"""Test eda.py trên dữ liệu giả (CPU): bảng lớp, ảnh sinh ra, và sanity check chạy thông với resnet18 nhỏ."""
import json
import tempfile
import unittest
from pathlib import Path

from helpers import make_fake_deepweeds
import eda


class TestEda(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.img, cls.lab, _ = make_fake_deepweeds(cls.root / "data", per_class=(12, 4, 4))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_eda_outputs(self):
        fig = self.root / "fig"
        s = eda.run_eda(self.img, self.lab, fig, n_per_class=3, expected_total=None)
        for f in ("class_distribution.png", "samples_per_class.png", "class_counts.csv", "split_check.json",
                  "eda_summary.json"):
            self.assertTrue((fig / f).exists(), f)
        self.assertEqual(len(s["class_table"]), 9)
        self.assertAlmostEqual(s["imbalance_ratio_train"], 1.0)

    def test_sanity_runs(self):
        fig = self.root / "fig2"
        o = eda.run_sanity("resnet18", self.img, self.lab, fig, img_size=64, batch_size=16, pretrained=False,
                           overfit_steps=40, device="cpu")
        self.assertTrue((fig / "augmentation_preview.png").exists())
        self.assertTrue((fig / "sanity_overfit.png").exists())
        self.assertLess(o["overfit_last_loss"], o["overfit_first_loss"])
        self.assertEqual(json.loads((fig / "sanity.json").read_text(encoding="utf-8"))["backbone"], "resnet18")


if __name__ == "__main__":
    unittest.main()
