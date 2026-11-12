"""INT8 matrix multiplication produces the same INT32 results as PyTorch.

Run: python scripts/checks/features/test_aten_int_mm.py -v
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


class Int8MatrixMultiplicationTests(unittest.TestCase):
    def test_signed_int8_matrix_multiplication(self):
        class Model(torch.nn.Module):
            def forward(self, x, weight):
                return torch._int_mm(x, weight)

        values = torch.arange(-32, 32, dtype=torch.int8).reshape(8, 8)
        model = Model()
        arguments = (values, values.t().contiguous())
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
