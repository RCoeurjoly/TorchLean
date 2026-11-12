"""Compiler tensor slicing preserves integer data and shape.

Run: python scripts/checks/features/test_pytorch_tensor_slicing.py -v
"""

import copy
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

import torch
import torchlean
from torchlean._fx import _tensor_value

CHECKER = os.environ.get("TORCHLEAN_VERIFY") or str(ROOT / ".lake/build/bin/verify")
CHECKS = ("structure", "types", "execute", "compare_bitwise")


class TensorSlicingTests(unittest.TestCase):
    def test_slice_first_column(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x[:, :1]

        model = Model()
        arguments = (torch.tensor([[3, 1, 4], [1, 5, 9]]),)
        with torch.no_grad():
            expected = copy.deepcopy(model)(*copy.deepcopy(arguments))
            torchlean.compile(model, checker=CHECKER)
            actual = model(*arguments)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                    expected.contiguous().view(torch.uint8)))

        # Compare the checked compiler IR itself; framework execution alone
        # cannot show that the imported computation has the right semantics.
        document = model.torchlean.artifact
        document["arguments"] = [_tensor_value(value) for value in arguments]
        document["reference_outputs"] = [_tensor_value(expected)]
        report = torchlean.verify(document, checker=CHECKER, checks=CHECKS)
        self.assertTrue(report["ok"], report)
        self.assertEqual({item["name"]: item["status"] for item in report["checks"]},
                         dict(structure="checked", types="checked", execute="executed",
                              compare_bitwise="compared"))


if __name__ == "__main__":
    unittest.main()
