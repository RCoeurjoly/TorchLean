"""TinyStories-1M with CPU dynamic INT8 linear layers and TorchLean capture.

Install dependencies in your venv: python -m pip install -e python torch transformers
Install CPU TorchAO: python -m pip install torchao --index-url https://download.pytorch.org/whl/cpu
Build the checker first: scripts/lake.sh build verify
TorchLean may reject unsupported embedding or quantized operations on the first call.
"""

import torch
import torchlean
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "roneneldan/TinyStories-1M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, attn_implementation="eager"
).cpu().eval()

# INT8 weights per output channel and dynamic INT8 activations per token.
# Embeddings, normalization, and attention stay floating point.
quantize_(model, Int8DynamicActivationInt8WeightConfig(version=2))
model = torchlean.compile(model)

inputs = tokenizer("Once upon a time", return_tensors="pt")
with torch.no_grad():
    prediction = model(**inputs, use_cache=False, return_dict=False)[0]

print(model.torchlean.report["ok"])
print("Logits shape:", prediction.shape)
