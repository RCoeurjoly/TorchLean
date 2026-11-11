# TinyStories-1M with TorchAO: current TorchLean gaps

Open [the standalone HTML ranking](TINYSTORIES_GAP_RANKING.html). Its complete machine-readable
evidence is [TINYSTORIES_TORCHAO_SURVEY_EVIDENCE.json](TINYSTORIES_TORCHAO_SURVEY_EVIDENCE.json).
This report is maintained on `docs/tinystories-torchao-gap-analysis`; the older PT2E report remains
on `docs/tinystories-survey-c2ff005b`.

## Exact target

- TorchLean runtime: `c2ff005b2fb2a36ca0f09c6218c7eeb9fd667751`, with unchanged runtime and dependency sources.
- `roneneldan/TinyStories-1M`, checkpoint revision `ac533fb8b4f69c71894bf96badfe11e6294d9fcf`.
- Both `config.json` and `pytorch_model.bin` are hash checked before loading. The weight hash is
  `07f9609ea882b8163ff3b23d40e2b82cb715d409631beb15c84b164f3877dae7`.
- Python 3.12, PyTorch `2.15.0+cpu`, TorchAO `0.18.0+cpu`, Transformers `5.22.2`.
- `Int8DynamicActivationInt8WeightConfig(version=2)`: symmetric per-output-channel INT8 weights,
  symmetric per-token dynamic INT8 activations, INT32 matrix multiplication and floating rescaling.
- Float32 CPU model, eager attention, no KV cache, one thread, input `int64[1,32]`. The exact token
  values and output contract are in the evidence. Embeddings, normalization and attention retain floating work.

This is a new quantized computation relative to clean-LLM2FPGA's pinned PT2E per-tensor affine
archive. Using the same base checkpoint does not make the quantized programs numerically identical.

## Findings and evidence boundaries

The exported ATen/TorchAO graph has **1,596 nodes**, including declarations and output, with
**1,469 operator nodes across 44 operator families**. Seven families have observed local refusals:

| ID | Exported contract | Refused / calls |
| --- | --- | ---: |
| A01 | `aten._int_mm.default` | 49 / 49 |
| A02 | `aten.add_.Tensor` | 24 / 24 |
| A03 | `export.access_subclass_inner_tensor.default` | 49 / 98 |
| A04 | `torchao.choose_qparams_affine.default` | 49 / 49 |
| A05 | `torchao.quantize_affine.default` | 49 / 49 |
| A06 | `_operator.getitem` (quantization tuple projections) | 98 / 98 |
| A07 | `aten.detach_.default` | 8 / 8 |

The public compiler inventory covers **484 raw Dynamo nodes**, expanded by the ordinary backend's
linear translation to **557 nodes**, including **438 operator nodes across 31 families**. Nine
operator forms are refused. They include functional embedding, live-parameter layer normalization,
tensor indexing, a dtype-changing cumulative sum, differences and tensor factories. They are ranked
in a separate F-table in the HTML; their counts are not added to the exported graph as one universal backlog.

There is also **F00**, a semantic representation gap affecting 49 quantized linear calls. Live
`Int8Tensor` weights advertise logical float32 dtype while storing INT8 payloads and scales. The
current compiler rewrites their functional linear calls to floating transpose/matmul IR, without
explicitly representing activation quantization, INT8 payload/scale inputs and INT32 accumulation.
A small real quantized linear example passes structure/type checking while exposing this floating
IR. That is recorded as `structure_checked_only`, never as quantized frontend equivalence.

Both real whole-model entry points refuse conversion: the exported graph at quantized tensor-field
access, the public compiler at functional embedding. ATen embedding itself is recognized and its
small native word-comparison fixture passes. This explains the discrepancy that motivated the new analysis.

The report preserves every source node, dependency, argument, result metadata and probe outcome.
Eager quantized execution, immediate export execution and save/load round-trip execution are
required to have identical output bytes on the recorded input before surveying the loaded graph.
Internal diagnostic failures are distinct from missing support. Bulk payload serialization is
omitted only in local rule/emitter probes; the separate ordinary conversion uses real parameters
and the unchanged importer. Custom TorchAO operators are inventoried as operators; their bodies
are not silently replaced by a supported primitive or a stub.

Tuple output metadata is not counted as successful projection support. Each projection is tested
against the actual current lowering rule, so the 98 quantization projections remain explicit gaps.
An independent code review caught the earlier inspector's metadata shortcut; a regression test
now prevents that shortcut from reporting these nodes as recognized.

Small native fixtures exercise embedding, integer matmul/clamp, integer-to-float casts, per-token
absolute maximum, ties-to-even rounding, the two TorchAO quantization operators and the observed
in-place operations. Expected eager words, native reports or exact failures are retained. Passing
fixtures cover those examples only. Full-model native execution, full-model bitwise agreement,
universal proofs, held-out language quality and FPGA speed/area/timing are not established.

## Reproduce

From this branch, prepare the Python environment and build the ordinary CPU checker:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.15.0+cpu' 'torchao==0.18.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'transformers==5.22.2'
python -m pip install -e python
scripts/lake.sh build verify
python scripts/checks/test_tinystories_torchao_survey.py
python scripts/checks/tinystories_torchao_survey.py \
  --snapshot roneneldan/TinyStories-1M \
  --checker "$PWD/.lake/build/bin/verify" \
  --out _out/tinystories/torchao-new-run
```

The remote model is downloaded at the pinned revision, restricted to the two hash-checked files.
An existing local directory with those exact files may be supplied instead. The output directory
must be new or empty. The script rejects runtime/dependency changes relative to the baseline and
external operator-table overrides. It saves the actual `model.pt2`, operator table, JSON evidence
and standalone HTML; the large exported model stays outside Git in `_out`.

The recorded local source is `/nix/store/h12qjfj7z8c7cx7rhykvdqxv1wllyrr6-graph-core-model-snapshot`.
That directory is optional for reproduction; it contains the same hash-checked upstream checkpoint.
Package import deprecations and legacy checkpoint mask-buffer notices are captured in the report.

Render an existing measured report without rerunning the model:

```bash
python scripts/checks/render_tinystories_torchao_survey.py \
  _out/tinystories/torchao-new-run/report.json \
  --output docs/TINYSTORIES_GAP_RANKING.html
```

The HTML's gap IDs belong to this new target. A-IDs rank exported contracts; F-IDs rank public
compiler work. They do not reuse the old PT2E L/E identifiers or claim that implementing a fixed
number of local rules will complete the full compiler and hardware pipeline.
