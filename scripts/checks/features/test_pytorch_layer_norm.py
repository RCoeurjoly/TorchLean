"""Compiled layer normalization retains live affine parameters.

Run: python scripts/checks/features/test_pytorch_layer_norm.py -v
"""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

import torch
import torchlean
from torchlean import _compiler
from torchlean._fx import _tensor_value

CHECKER = os.environ.get("TORCHLEAN_VERIFY") or str(ROOT / ".lake/build/bin/verify")
CHECKS = ("structure", "types", "execute", "compare_bitwise")


class LayerNormTests(unittest.TestCase):
    def test_live_affine_parameters(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(4)

            def forward(self, x):
                return self.norm(x)

        model = Model().eval()
        arguments = (torch.tensor([[1., -1., 1., -1.], [2., -2., 2., -2.]]),)
        recording = []
        backend = _compiler._torch_backend

        def record_backend(graph, examples):
            execute = backend(graph, examples)

            def run(*values):
                recording.append(values)
                return execute(*values)

            return run

        with patch.object(_compiler, "_torch_backend", record_backend):
            torchlean.compile(model, checker=CHECKER)
            for iteration in range(2):
                with torch.no_grad():
                    if iteration:
                        model.norm.weight.add_(.5)
                        model.norm.bias.add_(1.)
                    # forward bypasses the outer compiled entry for the oracle.
                    expected = model.forward(*arguments)
                    actual = model(*arguments)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                            expected.contiguous().view(torch.uint8)))
                self.assertTrue(recording, "No compiler invocation was recorded")
                document = model.torchlean.artifact
                document["arguments"] = [_tensor_value(value) for value in recording[-1]]
                document["reference_outputs"] = [_tensor_value(expected)]
                report = torchlean.verify(document, checker=CHECKER, checks=CHECKS)
                self.assertTrue(report["ok"], report)
                self.assertEqual({item["name"]: item["status"] for item in report["checks"]},
                                 dict(structure="checked", types="checked", execute="executed",
                                      compare_bitwise="compared"))


if __name__ == "__main__":
    unittest.main()
