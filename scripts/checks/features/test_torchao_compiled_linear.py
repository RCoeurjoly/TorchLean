"""Compiled TorchAO W8A8 linear retains integer semantics and matches native execution.

Run: python scripts/checks/features/test_torchao_compiled_linear.py -v
"""

import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

import torch
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
import torchlean
from torchlean import _compiler
from torchlean._fx import _tensor_value
from torchlean._table import op_table

CHECKER = os.environ.get("TORCHLEAN_VERIFY") or str(ROOT / ".lake/build/bin/verify")
CHECKS = ("structure", "types", "execute", "compare_bitwise")


class QuantizedLinearTests(unittest.TestCase):
    def compare_linear(self, model, values, *, quantized):
        # Record the real backend's inputs, without replacing its lowering or
        # execution. The native checker must replay this captured artifact.
        recording = []
        backend = _compiler._torch_backend

        def record_backend(graph, examples):
            execute = backend(graph, examples)

            def run(*inputs):
                recording.append(inputs)
                return execute(*inputs)

            return run

        with torch.no_grad():
            expected = copy.deepcopy(model)(values)
            with patch.object(_compiler, "_torch_backend", record_backend):
                torchlean.compile(model, checker=CHECKER)
                actual = model(values)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                    expected.contiguous().view(torch.uint8)))
        self.assertTrue(recording, "No compiler invocation was recorded")

        # Int8Tensor has logical float dtype and separate INT8 payload/scale.
        # Bind each real value to the artifact's declared shape and scalar type.
        # An all-floating artifact receives dequantized weights, which still
        # cannot reproduce dynamic activation quantization.
        candidates = []
        for value in recording[-1]:
            if hasattr(value, "qdata") and hasattr(value, "scale"):
                candidates.extend((value.qdata, value.scale, value.dequantize()))
            else:
                candidates.append(value)
        document = model.torchlean.artifact
        artifact = document["artifact"]
        inputs = sorted((node["input"]["index"], index, node["input"]["port"]["value"])
                        for index, node in enumerate(artifact["nodes"]) if "input" in node)
        dtypes = op_table(CHECKER)["dtypes"]
        arguments = []
        for _, node_id, shape in inputs:
            scalar = artifact["scalars"][node_id]
            for index, value in enumerate(candidates):
                dtype = str(value.dtype).removeprefix("torch.")
                if list(value.shape) == shape and dtypes[dtype] == scalar:
                    arguments.append(_tensor_value(candidates.pop(index)))
                    break
            else:
                self.fail(f"Cannot bind native IR input {shape}, {scalar}")
        document["arguments"] = arguments
        document["reference_outputs"] = [_tensor_value(expected)]
        report = torchlean.verify(document, checker=CHECKER, checks=CHECKS)
        self.assertTrue(report["ok"], report)
        self.assertEqual({item["name"]: item["status"] for item in report["checks"]},
                         dict(structure="checked", types="checked", execute="executed",
                              compare_bitwise="compared"))
        if quantized:
            integers = {scalar.get("width") for scalar in artifact["scalars"]
                        if isinstance(scalar, dict) and scalar.get("kind") == "integer"}
            self.assertTrue({8, 32} <= integers,
                            "W8A8 payloads and INT32 computation disappeared from the IR")

    def test_float_linear_control(self):
        # Dyadic values avoid irrelevant fused floating-point rounding changes.
        model = torch.nn.Sequential(torch.nn.Linear(8, 8)).eval()
        with torch.no_grad():
            model[0].weight.copy_(torch.arange(64.).reshape(8, 8) / 64)
            model[0].bias.copy_(torch.arange(8.) / 8)
        self.compare_linear(model, torch.arange(16.).reshape(2, 8) / 4, quantized=False)

    def test_dynamic_int8_activations_and_int8_weights(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = torch.nn.Sequential(torch.nn.Linear(8, 8)).eval()
            quantize_(model, Int8DynamicActivationInt8WeightConfig(version=2))
            values = torch.randn(2, 8)
        self.compare_linear(model, values, quantized=True)


if __name__ == "__main__":
    unittest.main()
