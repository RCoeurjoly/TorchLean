# TorchAO TinyStories gap analysis

User request: analyze the new per-token symmetric W8A8 TinyStories approach on a separate branch and publish a standalone HTML ranking like the earlier report.

## Target and evidence

- TorchLean runtime baseline: c2ff005b2fb2a36ca0f09c6218c7eeb9fd667751. Diagnostic/report changes must not change runtime support.
- Original TinyStories-1M checkpoint revision ac533fb8b4f69c71894bf96badfe11e6294d9fcf; CPU float32, eager attention, inference, no KV cache.
- TorchAO Int8DynamicActivationInt8WeightConfig(version=2), per-token symmetric activations and per-output-channel symmetric weights; record exact installed package versions and source/model/checker hashes.
- Primary graph: torch.export.export of the quantized logits module at int64[1,32]. Secondary path: public torchlean.compile on the same computation. Explicitly account for every source node in each inspected graph.
- Separate local rule/emitter recognition, complete conversion, native fixture execution and bitwise sample comparisons. No successful full-model or universal-proof claim unless actually established.

## Implementation and verification

1. Create docs/tinystories-torchao-gap-analysis; preserve the existing root example. Add diagnostic unit tests for coverage, diagnostic failures, and HTML escaping/status rendering; observe failures before implementation.
2. Implement a reproducible TorchAO model exporter and inventory using the current TorchLean rules, retaining dtype/shape/arguments/dependencies and frozen/runtime roles. Record first complete-conversion failures and the public compiler failure; keep inspecting after local refusals.
3. Run small real native fixtures for newly observed contracts and the ATen-vs-functional embedding discrepancy. Preserve failures as evidence rather than substituting stubs. Confirm eager exported execution matches the quantized model on the example.
4. Rank observed gaps with difficulty, actual signatures/call counts, implementation reuse, FPGA implications, and explicit limits. Generate docs/TINYSTORIES_GAP_RANKING.html, documentation, and a checked-in JSON evidence snapshot from the measured run.
5. Run diagnostic tests, the complete survey, JSON/HTML coverage consistency checks, syntax/whitespace checks, and independent review. Commit the diagnostic/report branch without merging or changing runtime behavior.

## Ledger

- Initial inspection: current main is the requested baseline; old report is on docs/tinystories-survey-c2ff005b. Runtime already has ATen embedding support but public compilation fails on functional embedding.
- Branch creation requires Git metadata permission because .git is mounted read-only in the sandbox; an authorized branch-creation request is pending. Independent survey preparation can proceed as untracked files.
- Branch created and checked out after the initial permission-review timeout; main and its runtime remain untouched.
- Initial diagnostic tests observed RED before the survey modules existed, then GREEN. Added a compiler-inventory regression after the adapter lacked export-only call_spec metadata; observed RED then GREEN.
- Independent review confirmed source counts and F00, and found two diagnostic problems: automatic acceptance of tuple projections and unexpected compiler errors labeled unsupported. Both have dedicated regressions observed RED then GREEN; tuple projections now use the actual current rule.
- F00 is a separately evidenced semantic representation gap, not a refused operator family. A real quantized linear fixture passes structure/types while its checked IR contains floating matmul and no explicit W8A8 quantizer.
- Final measurement: `_out/tinystories/torchao-20261111-final/report.json`; 1596 exported nodes / 44 operator families / 7 refused families, and 557 compiler-boundary nodes / 31 families / 9 refused forms. Both inventories have zero diagnostic errors; F00 affects 49 quantized linear calls.
- Exported model execution and serialized round-trip execution match eager quantized output bytes on the recorded input. Twelve native fixtures record five bitwise sample passes, six refusals and one structure-only success; no full-model native execution or proof is claimed.
- Verification: eight diagnostic tests passed; Python syntax, runtime/dependency immutability, all recorded file hashes, every source-node dependency/count and standalone HTML structure/anchors checked. Independent review's two important findings fixed by regressions observed RED then GREEN.
- Deliverables: standalone HTML, full JSON evidence, pinned reproduction documentation and diagnostic/test sources. Preserve the requested analysis branch; do not merge or push.
