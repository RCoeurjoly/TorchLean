"""In-place detach of a fresh frozen scalar preserves its payload.

Run: python scripts/checks/features/test_aten_detach_inplace.py -v
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


class InplaceDetachTests(unittest.TestCase):
    def test_frozen_negative_zero_scalar(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                scalar = torch.tensor(-0.)
                scalar.detach_()
                return x + scalar

        model = Model()
        arguments = (torch.tensor([-0., 0., 1., -1.]),)
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
