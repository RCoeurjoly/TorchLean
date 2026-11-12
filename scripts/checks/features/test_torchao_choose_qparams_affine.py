"""Per-token symmetric TorchAO calibration returns checked scale and zero point.

Run: python scripts/checks/features/test_torchao_choose_qparams_affine.py -v
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


class AffineCalibrationTests(unittest.TestCase):
    def test_per_token_symmetric_calibration(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ops.torchao.choose_qparams_affine.default(
                    x, "SYMMETRIC", [1, 1, 4], torch.int8, -128, 127,
                    torch.finfo(torch.float32).eps, torch.float32, torch.int8, True)

        model = Model()
        # Mixed signs, an all-zero token and values below the scale floor.
        arguments = (torch.tensor([[[-2., -1., 0., 1.], [0., 0., 0., 0.],
                                    [-1e-9, 0., 1e-9, 0.]]]),)
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
