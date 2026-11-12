"""In-place bias addition on a fresh tensor preserves PyTorch results.

Run: python scripts/checks/features/test_aten_add_inplace.py -v
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


class InplaceAdditionTests(unittest.TestCase):
    def test_fresh_tensor_bias_addition(self):
        class Model(torch.nn.Module):
            def forward(self, x, bias):
                fresh = x * 1.
                return fresh.add_(bias)

        model = Model()
        arguments = (torch.tensor([[-0., 0., 1., -1.], [2., -2., .5, -.5]]),
                     torch.tensor([-0., .5, 1., -2.]))
        # Invalid eager computations or exports must fail visibly.
        with torch.no_grad():
            model(*arguments)
        torch.export.export(model, arguments)
        report = torchlean.verify(model, *arguments, checker=CHECKER, checks=CHECKS)
        self.assertTrue(report["ok"], report)
        self.assertEqual({item["name"]: item["status"] for item in report["checks"]},
                         dict(structure="checked", types="checked", execute="executed",
                              compare_bitwise="compared"))


if __name__ == "__main__":
    unittest.main()
