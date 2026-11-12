"""Compiled embedding uses live weights and matches native lookup execution.

Run: python scripts/checks/features/test_pytorch_embedding.py -v
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


class EmbeddingTests(unittest.TestCase):
    def test_exported_embedding_matches_pytorch(self):
        model = torch.nn.Embedding(8, 4).eval()
        arguments = (torch.tensor([[0, 3, 7]]),)
        # Invalid eager computations or exports must fail visibly.
        with torch.no_grad():
            model(*arguments)
        torch.export.export(model, arguments)
        report = torchlean.verify(model, *arguments, checker=CHECKER, checks=CHECKS)
        self.assertTrue(report["ok"], report)
        self.assertEqual({item["name"]: item["status"] for item in report["checks"]},
                         dict(structure="checked", types="checked", execute="executed",
                              compare_bitwise="compared"))

    def test_compiled_embedding_keeps_weights_live(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 4)

            def forward(self, tokens):
                return self.embedding(tokens)

        model = Model().eval()
        arguments = (torch.tensor([[0, 3, 7]]),)
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
                        model.embedding.weight.add_(.5)
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
