"""Tuple projections preserve both outputs of a real quantization producer.

Run: python scripts/checks/features/test_pytorch_tuple_projection.py -v
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


class TupleProjectionTests(unittest.TestCase):
    # These cases also require choose_qparams_affine support. The producer must
    # remain real: replacing it with metadata would bypass the missing lowering.
    def compare_projection(self, component):
        class Model(torch.nn.Module):
            def forward(self, x):
                pair = torch.ops.torchao.choose_qparams_affine.default(
                    x, "SYMMETRIC", [1, 1, 4], torch.int8, -128, 127,
                    torch.finfo(torch.float32).eps, torch.float32, torch.int8, True)
                return pair[component]

        model = Model()
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

    def test_scale_projection(self):
        self.compare_projection(0)

    def test_zero_point_projection(self):
        self.compare_projection(1)


if __name__ == "__main__":
    unittest.main()
