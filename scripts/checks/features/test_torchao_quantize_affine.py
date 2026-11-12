"""TorchAO affine quantization preserves saturation and half-way rounding.

Run: python scripts/checks/features/test_torchao_quantize_affine.py -v
"""

import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

import torch
import torchao  # Registers the real custom quantization operators.
import torchlean

CHECKER = os.environ.get("TORCHLEAN_VERIFY") or str(ROOT / ".lake/build/bin/verify")
CHECKS = ("structure", "types", "execute", "compare_bitwise")


class AffineQuantizationTests(unittest.TestCase):
    def test_rounding_and_saturation(self):
        class Model(torch.nn.Module):
            def forward(self, x, scale, zero):
                return torch.ops.torchao.quantize_affine.default(
                    x, [1, 1, 4], scale, zero, torch.int8, -128, 127)

        model = Model()
        arguments = (torch.tensor([[[-1000., -64.25, -.25, .25],
                                     [127.5, 128., .5, 1.5]]]),
                     torch.tensor([[[.5], [1.]]]),
                     torch.zeros(1, 2, 1, dtype=torch.int8))
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
