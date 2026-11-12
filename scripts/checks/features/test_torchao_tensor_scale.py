"""Frozen TorchAO tensor-subclass scale fields remain available to export.

Run: python scripts/checks/features/test_torchao_tensor_scale.py -v
"""

import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

import torch
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
import torchlean

CHECKER = os.environ.get("TORCHLEAN_VERIFY") or str(ROOT / ".lake/build/bin/verify")
CHECKS = ("structure", "types", "execute", "compare_bitwise")


class TensorSubclassScaleTests(unittest.TestCase):
    def test_frozen_int8_weight_scale(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)

            def forward(self, x):
                return self.linear.weight.scale + x

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = Model().eval()
            quantize_(model, Int8DynamicActivationInt8WeightConfig(version=2))
        arguments = (torch.zeros(8, 1),)
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
