"""Test các thành phần lõi: losses, model, dataset, inference, benchmark (chạy trên CPU)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import make_fake_deepweeds
import benchmark as B
import dataset as D
import inference as I
import losses as L
import model as M


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.z = torch.randn(32, 9)
        self.y = torch.randint(0, 9, (32,))

    def test_focal_gamma0_equals_ce(self):
        ce = F.cross_entropy(self.z, self.y)
        fl = L.FocalLoss(gamma=0.0)(self.z, self.y)
        self.assertLess(abs(float(ce - fl)), 1e-6)

    def test_focal_reduces_easy_examples(self):
        self.assertLess(float(L.FocalLoss(2.0)(self.z, self.y)), float(F.cross_entropy(self.z, self.y)))

    def test_label_smoothing_matches_torch_and_eps0_is_ce(self):
        for eps in (0.0, 0.1, 0.3):
            mine = L.LabelSmoothingCE(eps)(self.z, self.y)
            ref = F.cross_entropy(self.z, self.y, label_smoothing=eps)
            self.assertLess(abs(float(mine - ref)), 1e-6, eps)

    def test_class_weights(self):
        counts = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 9000])
        w = L.class_weights(counts, 0.0)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)
        self.assertGreater(float(w[0]), float(w[8]))
        wb = L.class_weights(counts, 0.999)
        self.assertAlmostEqual(float(wb.sum()), 9.0, places=4)
        self.assertGreater(float(wb[0]), float(wb[8]))

    def test_weighted_ce_uniform_weights_equal_ce(self):
        crit = L.build_criterion("ce_weighted", weight=torch.ones(9))
        self.assertLess(abs(float(crit(self.z, self.y) - F.cross_entropy(self.z, self.y))), 1e-6)

    def test_cutmix_lambda_matches_pasted_area_and_labels(self):
        # ảnh i có giá trị hằng i, nên so sánh pixel cho biết chính xác vùng nào còn của ảnh gốc
        x = torch.arange(8, dtype=torch.float32).view(8, 1, 1, 1).expand(8, 3, 32, 32).clone()
        y = torch.arange(8)
        checked = 0
        for seed in range(30):
            rng = np.random.default_rng(seed)
            xm, (ya, yb, lam) = L.mix_batch(x, y, 1.0, "cutmix", rng=rng)
            self.assertTrue(torch.equal(ya, y))
            self.assertTrue(0.0 <= lam <= 1.0)
            for i in range(8):
                if int(yb[i]) != i:
                    frac_orig = float((xm[i, 0] == x[i, 0]).float().mean())
                    self.assertAlmostEqual(frac_orig, lam, places=5)
                    checked += 1
        self.assertGreater(checked, 50)

    def test_mixup_is_convex_and_does_not_modify_input(self):
        x = torch.rand(4, 3, 8, 8)
        x0 = x.clone()
        xm, (ya, yb, lam) = L.mix_batch(x, torch.arange(4), 1.0, "mixup", rng=np.random.default_rng(0))
        self.assertTrue(torch.equal(x, x0))
        self.assertLessEqual(float(xm.max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(xm.min()), 0.0)
        expected = lam * x + (1 - lam) * x[torch.tensor([int(v) for v in yb])]
        self.assertTrue(torch.allclose(xm, expected, atol=1e-6))

    def test_mixed_loss(self):
        crit = nn.CrossEntropyLoss()
        yb = torch.roll(self.y, 1)
        got = L.mixed_loss(crit, self.z, (self.y, yb, 0.3))
        exp = 0.3 * F.cross_entropy(self.z, self.y) + 0.7 * F.cross_entropy(self.z, yb)
        self.assertLess(abs(float(got - exp)), 1e-6)
        self.assertLess(abs(float(L.mixed_loss(crit, self.z, (self.y, yb, 1.0)) - F.cross_entropy(self.z, self.y))), 1e-6)

    def test_build_criterion_rejects_unknown(self):
        with self.assertRaises(ValueError):
            L.build_criterion("bogus")
        with self.assertRaises(ValueError):
            L.build_criterion("ce_weighted")


class TestModel(unittest.TestCase):
    def test_param_groups_three_groups_and_decay_rules(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        g = M.param_groups(m, 1e-4, 1e-3, 0.05)
        self.assertEqual([x["name"] for x in g], ["backbone_decay", "backbone_no_decay", "head"])
        by = {x["name"]: x for x in g}
        self.assertEqual(by["backbone_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(by["head"]["lr"], 1e-3)
        self.assertEqual(by["backbone_decay"]["lr"], 1e-4)
        self.assertTrue(all(p.ndim > 1 for p in by["backbone_decay"]["params"]))
        self.assertTrue(all(p.ndim <= 1 for p in by["backbone_no_decay"]["params"]))
        self.assertEqual(sum(len(x["params"]) for x in g), len(list(m.parameters())))

    def test_vit_pos_embed_not_decayed(self):
        m = M.build_model("vit_tiny_patch16_224", pretrained=False, init="scratch")
        no_decay_ids = {id(p) for p in M.param_groups(m, 1e-4, 1e-3, 0.05)[1]["params"]}
        self.assertIn(id(m.pos_embed), no_decay_ids)
        self.assertIn(id(m.cls_token), no_decay_ids)

    def test_freeze_backbone_trains_only_head_and_bn_stays_eval(self):
        m = M.build_model("resnet18", pretrained=False, init="finetune")
        M.freeze_backbone(m)
        head = {id(p) for p in m.get_classifier().parameters()}
        for p in m.parameters():
            self.assertEqual(p.requires_grad, id(p) in head)
        bn_before = m.bn1.running_mean.clone()
        M.set_train_mode(m)
        self.assertFalse(m.bn1.training)
        self.assertTrue(m.get_classifier().training)
        m(torch.randn(4, 3, 64, 64))
        self.assertTrue(torch.equal(bn_before, m.bn1.running_mean))
        self.assertEqual([x["name"] for x in M.param_groups(m, 1e-4, 1e-3, 0.05)], ["head"])

    def test_counts(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        self.assertAlmostEqual(M.count_params(m), 11.18, delta=0.05)
        self.assertAlmostEqual(M.count_gmacs(m, 224), 1.82, delta=0.1)

    def test_initial_loss_near_ln9(self):
        torch.manual_seed(0)
        m = M.build_model("resnet18", pretrained=False, init="scratch").eval()
        x, y = torch.randn(64, 3, 64, 64), torch.randint(0, 9, (64,))
        with torch.no_grad():
            loss = float(F.cross_entropy(m(x), y))
        self.assertLess(abs(loss - np.log(9)), 0.6)

    def test_invalid_init(self):
        with self.assertRaises(ValueError):
            M.build_model("resnet18", pretrained=False, init="bogus")

    def test_pinned_tags_exist_in_timm(self):
        import timm
        avail = set(timm.list_pretrained())
        for key, name in M.SUGGESTED_BACKBONES.items():
            self.assertIn(name, avail, key)


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.img, cls.lab, cls.dfs = make_fake_deepweeds(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_load_and_check_split(self):
        tr, va, te = D.load_split(self.lab, 0)
        res = D.check_split(tr, va, te, self.img, expected_total=None, verbose=False)
        self.assertEqual(res["n"], {"train": 108, "val": 36, "test": 36})
        self.assertEqual(res["overlap"], {"train&val": 0, "train&test": 0, "val&test": 0})

    def test_check_split_detects_leak_missing_total_and_ratio(self):
        tr, va, te = D.load_split(self.lab, 0)
        with self.assertRaisesRegex(ValueError, "giao"):
            D.check_split(pd.concat([tr, va.iloc[:1]]), va, te, self.img, expected_total=None, verbose=False)
        with self.assertRaisesRegex(ValueError, "hợp ba tập"):
            D.check_split(tr, va, te, self.img, expected_total=17509, verbose=False)
        with self.assertRaisesRegex(ValueError, "tỉ lệ"):
            D.check_split(tr.iloc[:50], va, te, self.img, expected_total=None, verbose=False)
        bad = va.copy()
        bad.loc[bad.index[0], "Filename"] = "khong_ton_tai.jpg"
        with self.assertRaisesRegex(ValueError, "không tồn tại"):
            D.check_split(tr, bad, te, self.img, expected_total=None, verbose=False)

    def test_transforms_shapes_and_eval_deterministic(self):
        tr, _, _ = D.load_split(self.lab, 0)
        ds_train = D.DeepWeedsDataset(tr, self.img, D.build_transforms(True, 224, "basic"))
        x, y, f = ds_train[0]
        self.assertEqual(tuple(x.shape), (3, 224, 224))
        self.assertIsInstance(y, int)
        self.assertTrue(f.endswith(".jpg"))
        ds_eval = D.DeepWeedsDataset(tr, self.img, D.build_transforms(False, 224))
        self.assertTrue(torch.equal(ds_eval[3][0], ds_eval[3][0]))
        full = D.DeepWeedsDataset(tr, self.img, D.build_transforms(False, 224, test_size=256))
        self.assertEqual(tuple(full[0][0].shape), (3, 256, 256))
        # view "center" từ ảnh đầy đủ phải trùng đúng với transform eval chuẩn (mốc I00)
        center = full[3][0][:, 16:240, 16:240]
        self.assertTrue(torch.allclose(center, ds_eval[3][0], atol=1e-6))

    def test_all_aug_modes_run(self):
        tr, _, _ = D.load_split(self.lab, 0)
        for aug in D.AUG_CHOICES:
            ds = D.DeepWeedsDataset(tr.iloc[:3], self.img, D.build_transforms(True, 64, aug))
            self.assertEqual(tuple(ds[0][0].shape), (3, 64, 64), aug)
        with self.assertRaises(ValueError):
            D.build_transforms(True, 64, "nope")

    def test_eval_loader_keeps_order_and_balanced_sampler_balances(self):
        tr, va, _ = D.load_split(self.lab, 0)
        loader = D.make_loader(va, self.img, D.build_transforms(False, 64), 8, False, num_workers=0)
        got = [f for _, _, fs in loader for f in fs]
        self.assertEqual(got, va["Filename"].tolist())
        skew = pd.concat([tr, tr[tr.Label == 8], tr[tr.Label == 8], tr[tr.Label == 8]])  # lệch về lớp 8
        loader = D.make_loader(skew, self.img, D.build_transforms(False, 32), 16, True, sampler="balanced",
                               num_workers=0)
        counts = np.zeros(9)
        for _ in range(8):
            for _, y, _ in loader:
                for c in y.tolist():
                    counts[c] += 1
        self.assertLess(abs(counts[8] / counts.sum() - 1 / 9), 0.06)

    def test_denormalize_roundtrip(self):
        x = torch.rand(3, 8, 8)
        n = (x - torch.tensor(D.IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(D.IMAGENET_STD).view(3, 1, 1)
        self.assertTrue(torch.allclose(D.denormalize(n), x, atol=1e-6))


class TestInference(unittest.TestCase):
    def test_aggregate_and_softmax(self):
        rng = np.random.default_rng(0)
        z1, z2 = rng.normal(size=(20, 9)), rng.normal(size=(20, 9))
        for space in ("prob", "logit"):
            np.testing.assert_allclose(I.aggregate_views([z1, z2], space).sum(1), 1.0, atol=1e-9)
        np.testing.assert_allclose(I.aggregate_views([z1], "prob"), I.softmax_np(z1))
        self.assertFalse(np.allclose(I.aggregate_views([z1, z2], "prob"), I.aggregate_views([z1, z2], "logit")))
        with self.assertRaises(ValueError):
            I.aggregate_views([z1], "bogus")

    def test_ensemble(self):
        a, b = np.full((4, 9), 1 / 9), np.eye(9)[:4]
        np.testing.assert_allclose(I.ensemble_probs([a, b]), (a + b) / 2)
        with self.assertRaises(ValueError):
            I.ensemble_probs([a, np.ones((5, 9)) / 9])

    def test_temperature_recovers_known_T_and_keeps_accuracy(self):
        rng = np.random.default_rng(1)
        n, k, true_T = 6000, 9, 2.5
        logits = rng.normal(size=(n, k)) * 3
        probs = I.softmax_np(logits / true_T)
        y = np.array([rng.choice(k, p=p) for p in probs])  # nhãn sinh đúng từ p = softmax(logit / T)
        T = I.fit_temperature(logits, y)
        self.assertAlmostEqual(T, true_T, delta=0.25)

        def nll(p):
            return -np.log(p[np.arange(n), y]).mean()

        self.assertLess(nll(I.apply_temperature(logits, T)), nll(I.softmax_np(logits)))
        np.testing.assert_array_equal(I.apply_temperature(logits, T).argmax(1), logits.argmax(1))

    def test_temperature_views_overconfident_gives_T_above_1(self):
        rng = np.random.default_rng(2)
        z = [rng.normal(size=(500, 9)) * 4 for _ in range(2)]
        y = rng.integers(0, 9, 500)  # nhãn ngẫu nhiên nên mô hình luôn quá tự tin
        for space in ("prob", "logit"):
            self.assertGreater(I.fit_temperature_views(z, y, space), 1.0)

    def test_views_shapes(self):
        x = torch.randn(2, 3, 256, 256)
        v = I.standard_views(224)
        self.assertEqual(tuple(v["center"](x).shape), (2, 3, 224, 224))
        self.assertTrue(torch.equal(v["center_flip"](x), torch.flip(v["center"](x), dims=[-1])))
        self.assertEqual(tuple(v["full288"](x).shape), (2, 3, 288, 288))
        self.assertEqual(len(I.views_multicrop(x, 224)), 5)
        self.assertEqual(len(I.views_multicrop(x, 224, flip=True)), 10)
        self.assertTrue(torch.equal(I.view_hflip(I.view_hflip(x)), x))
        for names in I.VIEW_SETS.values():
            for nm in names:
                self.assertIn(nm, v)

    def test_fuse_conv_bn_resnet_mobilenet_and_no_bn(self):
        import timm
        for name, expect_fused in (("resnet18", True), ("mobilenetv3_small_050", True), ("convnext_atto", False)):
            torch.manual_seed(0)
            m = timm.create_model(name, pretrained=False, num_classes=9).eval()
            for mod in m.modules():  # BN có thống kê khác mặc định để phép thử có ý nghĩa
                if isinstance(mod, nn.BatchNorm2d):
                    mod.running_mean.normal_(0, 0.3)
                    mod.running_var.uniform_(0.5, 2.0)
                    mod.weight.data.uniform_(0.5, 1.5)
                    mod.bias.data.normal_(0, 0.3)
            f = I.fuse_conv_bn(m, verbose=False)
            self.assertEqual(f.fuse_info["n_fused"] > 0, expect_fused, name)
            self.assertLess(f.fuse_info["max_abs_err"], 1e-3, name)
            x = torch.randn(2, 3, 64, 64)
            with torch.no_grad():
                self.assertLess(float((m(x) - f(x)).abs().max()), 1e-3)
            if expect_fused:  # mọi BN đã biến mất ở bản gộp, và model gốc không bị sửa
                self.assertEqual(sum(isinstance(mod, nn.BatchNorm2d) for mod in f.modules()), 0, name)
                self.assertTrue(any(isinstance(mod, nn.BatchNorm2d) for mod in m.modules()))

    def test_predict_views_matches_direct_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            img, lab, _ = make_fake_deepweeds(Path(tmp), per_class=(2, 2, 2))
            _, va, _ = D.load_split(lab, 0)
            loader = D.make_loader(va, img, D.build_transforms(False, 64, test_size=64), 8, False, num_workers=0)
            m = M.build_model("resnet18", pretrained=False, init="scratch").eval()
            names, y, out = I.predict_views(m, loader, "cpu", {"a": I.view_identity, "b": I.view_hflip})
            self.assertEqual(names, va["Filename"].tolist())
            x, _, _ = next(iter(loader))
            with torch.no_grad():
                np.testing.assert_allclose(out["a"][:8], m(x).numpy(), atol=1e-4)
                np.testing.assert_allclose(out["b"][:8], m(torch.flip(x, [-1])).numpy(), atol=1e-4)


class TestBenchmark(unittest.TestCase):
    def test_bench_percentiles_and_sync_calls(self):
        calls = {"sync": 0, "fn": 0}
        r = B.bench(lambda: calls.__setitem__("fn", calls["fn"] + 1), warmup=10, iters=60,
                    sync=lambda: calls.__setitem__("sync", calls["sync"] + 1))
        self.assertEqual(calls["fn"], 70)
        self.assertGreaterEqual(calls["sync"], 2 * 60)  # sync trước và sau mỗi lần đo
        self.assertLessEqual(r["p50"], r["p95"])
        self.assertLessEqual(r["p95"], r["p99"])
        self.assertEqual(r["n"], 60)

    def test_strict_rejects_too_few_iters_or_warmup(self):
        with self.assertRaises(ValueError):
            B.bench(lambda: None, warmup=5, iters=100)
        with self.assertRaises(ValueError):
            B.bench(lambda: None, warmup=10, iters=20)

    def test_latency_report_keys_on_cpu(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        r = B.latency_report(m, 1, 64, "fp32", "cpu", warmup=2, iters=5, strict=False)
        for k in ("gpu", "dtype", "batch", "img_size", "p50", "p95", "p99", "images_per_s", "torch"):
            self.assertIn(k, r)
        self.assertGreater(r["images_per_s"], 0)
        t = B.tta_latency(m, 2, batch_size=1, img_size=64, dtype="fp32", device="cpu", warmup=2, iters=5, strict=False)
        self.assertEqual(t["k_views"], 2)
        self.assertGreater(t["ratio_vs_single"], 1.0)


if __name__ == "__main__":
    unittest.main()
