"""Core ATen advanced indexing supports Boolean data and INT64 indices.

Run: python scripts/checks/features/test_pytorch_bool_advanced_index.py -v
"""

import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

import torch
import torchlean

CHECKER = os.environ.get("TORCHLEAN_VERIFY") or str(ROOT / ".lake/build/bin/verify")
CHECKS = ("structure", "types", "execute", "compare_bitwise")


class BooleanAdvancedIndexTests(unittest.TestCase):
    def assert_compares(self, model, *arguments):
        with torch.no_grad():
            expected = model(*arguments)
            core = torch.export.export(model, arguments, strict=False).run_decompositions()
            actual = core.module()(*arguments)
            self.assertEqual(actual.dtype, torch.bool)
            self.assertTrue(torch.equal(actual, expected))
        report = torchlean.verify(core.module(), *arguments, checker=CHECKER, checks=CHECKS)
        self.assertTrue(report["ok"], report)
        self.assertEqual({item["name"]: item["status"] for item in report["checks"]},
                         dict(structure="checked", types="checked", execute="executed",
                              compare_bitwise="compared"))

    def test_boolean_data_with_broadcast_integer_indices(self):
        class Model(torch.nn.Module):
            def forward(self, mask):
                # Boolean DATA is indexed by integer coordinates. This is not
                # Boolean-mask indexing with a data-dependent output shape.
                batch = torch.zeros((1, 1, 1, 1), dtype=torch.int64)
                positions = torch.arange(mask.shape[-1]).reshape(1, 1, 1, -1)
                return torch.ops.aten.index.Tensor(mask.bool(), [batch, positions])

        mask = torch.tensor([[1] * 24 + [0] * 8], dtype=torch.int64)
        self.assert_compares(Model(), mask)

    def test_runtime_indices_broadcast_wrap_and_preserve_axis_order(self):
        data = torch.arange(30).reshape(2, 3, 5) % 7 < 3
        for adjacent in (False, True):
            class Model(torch.nn.Module):
                def forward(self, x, rows, columns):
                    indices = [None, rows, columns] if adjacent else [rows, None, columns]
                    return torch.ops.aten.index.Tensor(x, indices)

            for dtype in (torch.int32, torch.int64):
                with self.subTest(adjacent_axes=adjacent, index_dtype=dtype):
                    rows = torch.tensor([[0], [-1]], dtype=dtype)
                    columns = torch.tensor([[-1, 1, 4]], dtype=dtype)
                    self.assert_compares(Model(), data, rows, columns)

    def test_boolean_mask_indices_remain_unsupported(self):
        class Model(torch.nn.Module):
            def forward(self, x, mask):
                return torch.ops.aten.index.Tensor(x, [mask])

        data = torch.tensor([True, False, False, True])
        mask = torch.tensor([True, False, True, False])
        report = torchlean.verify(Model(), data, mask, checker=CHECKER, checks=("structure",))
        self.assertFalse(report["ok"])
        self.assertEqual(report["checks"][0]["status"], "unsupported")
        self.assertIn("Boolean mask index", report["checks"][0]["message"])

    def test_boolean_index_put_remains_unsupported(self):
        data = torch.tensor([True, False, False, True])
        indices = torch.tensor([0, 2], dtype=torch.int64)
        values = torch.tensor([False, True])
        for accumulate in (False, True):
            class Model(torch.nn.Module):
                def forward(self, x, index, updates):
                    return torch.ops.aten.index_put.default(x, [index], updates, accumulate)

            with self.subTest(accumulate=accumulate):
                report = torchlean.verify(Model(), data, indices, values,
                                          checker=CHECKER, checks=("structure",))
                self.assertFalse(report["ok"])
                self.assertEqual(report["checks"][0]["status"], "unsupported")
                self.assertIn("of a bool tensor is not imported", report["checks"][0]["message"])


if __name__ == "__main__":
    unittest.main()
