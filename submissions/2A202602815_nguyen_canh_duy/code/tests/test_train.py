"""Test vòng huấn luyện: scheduler, EMA, parse_overrides, run() end-to-end trên dữ liệu giả (CPU)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from helpers import make_fake_deepweeds
import dataset as D
import inference as I
import model as M
import train as TR
from evalref import ev


def small_cfg(root: Path, **kw) -> TR.Config:
    base = dict(exp_id="X00", backbone="resnet18", pretrained=False, img_size=64, epochs=2, batch_size=16,
                num_workers=0, amp=False, measure_latency=False, expected_total=None,
                images_dir=str(root / "images"), labels_dir=str(root / "labels"),
                out_dir=str(root / "runs"), pred_dir=str(root / "predictions"), curves_dir=str(root / "curves"),
                lr_backbone=1e-3, lr_head=3e-3)
    base.update(kw)
    return TR.Config(**base)


class TestSchedulerEmaCli(unittest.TestCase):
    def test_lr_factor_warmup_then_cosine(self):
        total, warm = 100, 10
        f = [TR.lr_factor(s, total, warm) for s in range(total)]
        self.assertLess(f[0], 0.2)
        self.assertAlmostEqual(f[warm - 1], 1.0)
        self.assertAlmostEqual(f[warm], 1.0, places=2)
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(f[warm:], f[warm + 1:])))
        self.assertLess(f[-1], 0.01)
        self.assertEqual(TR.lr_factor(0, 100, 0), 1.0)  # không warmup: bắt đầu từ LR đầy đủ

    def test_scheduler_drives_param_group_lr(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        cfg = TR.Config(epochs=2, warmup_epochs=1.0, lr_backbone=1e-4, lr_head=1e-3)
        opt = TR.build_optimizer(m, cfg)
        sch = TR.build_scheduler(opt, cfg, steps_per_epoch=5)
        lrs = []
        for _ in range(10):
            opt.step()
            sch.step()
            lrs.append([g["lr"] for g in opt.param_groups])
        head = [l[-1] for l in lrs]
        self.assertAlmostEqual(max(head), 1e-3, delta=1e-4)  # đỉnh = LR head
        self.assertAlmostEqual(opt.param_groups[0]["initial_lr"], 1e-4)
        self.assertAlmostEqual(opt.param_groups[-1]["initial_lr"] / opt.param_groups[0]["initial_lr"], 10.0)

    def test_weight_decay_zero_for_norm_in_optimizer(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        opt = TR.build_optimizer(m, TR.Config())
        wd = {g["name"]: g["weight_decay"] for g in opt.param_groups}
        self.assertEqual(wd, {"backbone_decay": 0.05, "backbone_no_decay": 0.0, "head": 0.05})

    def test_ema_matches_manual_update_and_copies_int_buffers(self):
        torch.manual_seed(0)
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        ema = TR.EMA(m, 0.9)
        w0 = {k: v.clone() for k, v in m.state_dict().items()}
        with torch.no_grad():
            for p in m.parameters():
                p.add_(1.0)
        ema.update(m)
        d = min(0.9, 2 / 11)  # hâm nóng: (1+n)/(10+n) với n = 1
        for (k, v), p in zip(ema.module.state_dict().items(), m.state_dict().values()):
            if v.dtype.is_floating_point:
                torch.testing.assert_close(v, d * w0[k] + (1 - d) * p, atol=1e-5, rtol=1e-5)
        self.assertFalse(any(p.requires_grad for p in ema.module.parameters()))

    def test_parse_overrides_types(self):
        o = TR.parse_overrides(["seed=3", "loss=focal", "ema_decay=none", "lr_head=0.002", "amp=false",
                                "sampler=balanced", "limit_batches=2", "class_weight_beta=0.99"])
        self.assertEqual(o, {"seed": 3, "loss": "focal", "ema_decay": None, "lr_head": 0.002, "amp": False,
                             "sampler": "balanced", "limit_batches": 2, "class_weight_beta": 0.99})
        self.assertEqual(TR.Config(**o).seed, 3)
        with self.assertRaises(ValueError):
            TR.parse_overrides(["khong_co=1"])
        with self.assertRaises(ValueError):
            TR.parse_overrides(["seed"])
        with self.assertRaises(ValueError):
            TR.parse_overrides(["amp=maybe"])

    def test_paths_follow_eval_naming(self):
        cfg = TR.Config(exp_id="F01", seed=2)
        self.assertEqual(TR.pred_path(cfg, "test"), Path("predictions/F01_seed2_test.csv"))
        self.assertEqual(ev.parse_seed(str(TR.pred_path(cfg, "test"))), 2)
        self.assertEqual(TR.curve_path(TR.Config(exp_id="B01", backbone="resnet50")), Path("curves/B01_resnet50.png"))
        self.assertEqual(TR.curve_path(TR.Config(exp_id="B01", backbone="resnet50", seed=1, desc="x")),
                         Path("curves/B01_x_seed1.png"))


class TestRunEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        make_fake_deepweeds(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_basic_run_outputs_and_metric_consistency(self):
        cfg = small_cfg(self.root, exp_id="X01", epochs=3, measure_latency=True)
        s = TR.run(cfg)
        rd = TR.run_dir(cfg)
        for f in ("config.json", "history.csv", "best.pt", "summary.json", "val_logits.npz"):
            self.assertTrue((rd / f).exists(), f)
        self.assertFalse((rd / "last.pt").exists())
        self.assertTrue(TR.curve_path(cfg).exists())
        self.assertFalse((rd / "test_started.json").exists())  # mặc định KHÔNG đụng test
        self.assertFalse(list(Path(cfg.pred_dir).glob("*_test.csv")))
        h = pd.read_csv(rd / "history.csv")
        self.assertEqual(len(h), 3)
        self.assertLess(h["train_loss"].iloc[-1], h["train_loss"].iloc[0])
        # best epoch = macro-F1 val cao nhất (hòa thì sớm hơn); summary khớp history
        self.assertEqual(s["best_epoch"], int(np.argmax(h["val_macro_f1"].values)) + 1)
        z = np.load(rd / "val_logits.npz")
        p = I.softmax_np(z["logits"])
        m = ev.compute_metrics(z["y"], p.argmax(1), p)
        self.assertAlmostEqual(m["macro_f1"], s["val_macro_f1"], places=9)
        self.assertEqual(len(s["val_f1_per_class"]), 9)
        self.assertGreater(s["params_M"], 11)
        self.assertIsNotNone(s["latency_b1_ms"])
        # chạy lại -> bỏ qua vì đã có summary
        self.assertEqual(TR.run(cfg)["val_macro_f1"], s["val_macro_f1"])

    def test_load_run_model_reproduces_saved_val_logits(self):
        cfg = small_cfg(self.root, exp_id="X02", epochs=2)
        TR.run(cfg)
        rd = TR.run_dir(cfg)
        model, c2 = TR.load_run_model(rd)
        _, va, _ = D.load_split(cfg.labels_dir, 0)
        mean, std = TR._train_mean_std(model)
        loader = D.make_loader(va, cfg.images_dir, D.build_transforms(False, 64, mean=mean, std=std), 16, False,
                               num_workers=0)
        names, y, logits = I.predict_logits(model, loader, "cpu")
        z = np.load(rd / "val_logits.npz")
        self.assertEqual(names, z["files"].tolist())
        np.testing.assert_allclose(logits, z["logits"], atol=1e-4)

    def test_resume_after_interruption_completes_all_epochs(self):
        cfg = small_cfg(self.root, exp_id="X03", epochs=3)
        orig = TR.train_one_epoch
        calls = {"n": 0}

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt("giả lập phiên Colab bị ngắt")
            return orig(*a, **kw)

        with mock.patch.object(TR, "train_one_epoch", flaky):
            with self.assertRaises(KeyboardInterrupt):
                TR.run(cfg)
        rd = TR.run_dir(cfg)
        self.assertTrue((rd / "last.pt").exists())
        self.assertFalse((rd / "summary.json").exists())
        s = TR.run(cfg)
        h = pd.read_csv(rd / "history.csv")
        self.assertEqual(h["epoch"].tolist(), [0, 1, 2])
        self.assertEqual(s["best_epoch"], int(np.argmax(h["val_macro_f1"].values)) + 1)

    def test_variants_run(self):
        variants = {
            "ema": dict(ema_decay=0.9),
            "frozen": dict(init="frozen"),
            "cutmix": dict(mix="cutmix", mix_prob=1.0),
            "mixup": dict(mix="mixup", mix_prob=0.5),
            "ls": dict(loss="ls", label_smoothing=0.1),
            "focal": dict(loss="focal"),
            "weighted": dict(loss="ce_weighted", class_weight_beta=0.0),
            "balanced": dict(sampler="balanced"),
            "sgd": dict(optimizer="sgd", lr_backbone=0.01, lr_head=0.1, weight_decay=5e-4),
            "color_nowarm": dict(aug="color", warmup_epochs=0),
            "clip_limit": dict(grad_clip=1.0, limit_batches=2),
        }
        for i, (name, kw) in enumerate(variants.items()):
            cfg = small_cfg(self.root, exp_id=f"V{i:02d}", epochs=2, **kw)
            s = TR.run(cfg)
            self.assertTrue(np.isfinite(s["val_macro_f1"]), name)
            if name == "ema":
                self.assertIsNotNone(s["val_macro_f1_raw"])
        # frozen: backbone không đổi so với khởi tạo, chỉ head học
        cfg = small_cfg(self.root, exp_id="X04", init="frozen", epochs=2, seed=5)
        TR.set_seed(5)
        ref = M.build_model("resnet18", pretrained=False, init="frozen")
        before = {k: v.clone() for k, v in ref.state_dict().items()}
        TR.run(cfg)
        model, _ = TR.load_run_model(TR.run_dir(cfg))
        same = [torch.equal(before[k], v) for k, v in model.state_dict().items() if k.startswith("layer1")]
        self.assertTrue(all(same))

    def test_bad_mix_raises(self):
        with self.assertRaises(ValueError):
            TR.run(small_cfg(self.root, exp_id="X05", mix="bogus"))

    def test_test_guard_blocks_second_access(self):
        cfg = small_cfg(self.root, exp_id="X06", epochs=1, save_test_predictions=True, seed=7)
        TR.run(cfg)
        rd = TR.run_dir(cfg)
        self.assertTrue((rd / "test_done.json").exists())
        g = ev.load_group(str(TR.pred_path(cfg, "test")), str(Path(cfg.labels_dir) / "test_subset0.csv"))
        self.assertEqual(g.metrics[0]["n"], 36)
        with self.assertRaisesRegex(RuntimeError, "ĐÃ được mở"):
            TR.begin_test(rd)
        cfg2 = small_cfg(self.root, exp_id="X06", epochs=1, save_test_predictions=True, seed=7, overwrite=True)
        with self.assertRaisesRegex(RuntimeError, "ĐÃ được mở"):
            TR.run(cfg2)

    def test_alias_run(self):
        cfg = small_cfg(self.root, exp_id="X07", epochs=1)
        s = TR.run(cfg)
        a = TR.alias_run("X07", "X08", 0, out_dir=cfg.out_dir, curves_dir=cfg.curves_dir, desc="alias")
        self.assertEqual(a["exp_id"], "X08")
        self.assertEqual(a["alias_of"], "X07")
        self.assertEqual(a["val_macro_f1"], s["val_macro_f1"])
        self.assertTrue((Path(cfg.out_dir) / "X08" / "seed0" / "best.pt").exists())
        self.assertTrue((Path(cfg.curves_dir) / "X08_alias.png").exists())
        cfg_json = json.loads((Path(cfg.out_dir) / "X08" / "seed0" / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg_json["exp_id"], "X08")


if __name__ == "__main__":
    unittest.main()
