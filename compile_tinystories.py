"""Export quantized TinyStories-1M to Core ATen, then try TorchLean verification.

Install dependencies in your venv: python -m pip install -e python torch transformers
Install CPU TorchAO: python -m pip install torchao --index-url https://download.pytorch.org/whl/cpu
Build the checker first: scripts/lake.sh build verify
The .pt2 file is written even if TorchLean does not yet support every operator.
"""

from pathlib import Path

import torch
import torchlean
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
from transformers import AutoModelForCausalLM, AutoTokenizer


def int_mm_to_core(left, right):
    # Widen before multiplication: INT8 mm alone would return INT8 results.
    return torch.ops.aten.mm.default(left.to(torch.int32), right.to(torch.int32))


model_id = "roneneldan/TinyStories-1M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, attn_implementation="eager"
).cpu().eval()

# INT8 weights per output channel and dynamic INT8 activations per token.
# Embeddings, normalization, and attention stay floating point.
quantize_(model, Int8DynamicActivationInt8WeightConfig(version=2))

inputs = tokenizer("Once upon a time", return_tensors="pt")
arguments = {**inputs, "use_cache": False, "return_dict": False}
with torch.no_grad():
    expected = model(**arguments)[0]
    exported = torch.export.export(model, (), kwargs=arguments, strict=False)
    decompositions = torch.export.default_decompositions()
    decompositions[torch.ops.aten._int_mm.default] = int_mm_to_core
    core = exported.run_decompositions(decompositions)
    non_core = {str(node.target) for node in core.graph.nodes
                if node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload)
                and torch.Tag.core not in node.target.tags
                and node.target != torch.ops.aten._assert_tensor_metadata.default}
    assert not non_core, f"Unexpected non-Core numerical operators: {non_core}"
    prediction = core.module()(**arguments)[0]
    assert torch.equal(prediction.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8))

output = Path("_out/tinystories-core.pt2")
output.parent.mkdir(parents=True, exist_ok=True)
torch.export.save(core, output)
print("Saved:", output)
print("Logits shape:", prediction.shape)
print("Quantized eager/Core ATen output bytes match")

# verify accepts an nn.Module and re-exports it. The module supplied here already
# contains Core ATen operations. The exact exported graph is inventoried by
# scripts/checks/tinystories_core_aten_gaps.py without a second export.
report = torchlean.verify(core.module(), example_kwargs=arguments,
                          checks=("structure", "types"))
print("TorchLean supported:", report["ok"])
if not report["ok"]:
    for message in dict.fromkeys(check.get("message", "") for check in report["checks"]):
        print(message)
