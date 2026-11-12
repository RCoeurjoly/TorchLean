"""Measure TinyStories gaps after documented export and Core ATen decomposition.

Requires a local TinyStories-1M snapshot, torch, torchao, transformers and verify.
No TorchLean implementation is changed, no operator is stubbed in the main graph,
and the supplied checkpoint is never downloaded or overwritten.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import math
import operator
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
TOKENS = [7454, 2402, 257, 640, 11, 257, 1310, 2576, 3706, 20037, 1043, 257,
          6016, 2266, 2613, 287, 262, 11376, 13, 198, 3347, 3751, 340, 284, 607,
          3956, 11, 290, 1978, 484, 11686, 340]
CHECKS = ("structure", "types", "execute", "compare_bitwise")


def int_mm_to_core(left, right):
    import torch
    return torch.ops.aten.mm.default(left.to(torch.int32), right.to(torch.int32))


def core_decompositions():
    import torch
    table = torch.export.default_decompositions()
    table[torch.ops.aten._int_mm.default] = int_mm_to_core
    return table


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes():
    # Editors may create dangling .#foo.py lock symlinks alongside source files.
    return {str(path.relative_to(ROOT)): digest(path)
            for path in (ROOT / "python/torchlean").glob("*.py")
            if not path.name.startswith(".") and path.is_file()}


def describe(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def error_record(error):
    result = {"status": "unsupported" if isinstance(error, NotImplementedError) else "error",
              "exception": type(error).__name__, "message": str(error)}
    trace = error.__traceback__
    while trace:
        node = trace.tb_frame.f_locals.get("node")
        if hasattr(node, "name") and hasattr(node, "target"):
            result["source_node"] = node.name
            result["operator"] = str(node.target)
        trace = trace.tb_next
    return result


def bindings(program, arguments):
    values, names, supplied = {}, [], iter(arguments)
    for spec in program.graph_signature.input_specs:
        if spec.kind.name == "USER_INPUT":
            values[spec.arg.name] = next(supplied)
            names.append(spec.arg.name)
        else:
            values[spec.arg.name] = program.state_dict.get(
                spec.target, program.constants.get(spec.target))
    assert next(supplied, None) is None
    return values, names


@contextmanager
def capture_state(program, checker):
    from torchlean import _fx, _graph, _structure
    dictionaries = [(_fx, "_TENSOR_VALUES")] + [(_structure, name) for name in
        ("_FOLDED", "_BUFFERS", "_OUTPUT_TARGETS", "_BUFFER_UPDATES", "_DROPOUT_MASKS")]
    with _graph._LOCK:
        saved = [(module, name, copy.copy(getattr(module, name))) for module, name in dictionaries]
        previous_checker = _fx._CHECKER[:]
        try:
            _graph._reset(program.graph_signature, str(checker))
            state, _ = bindings(program, (None,) * sum(
                spec.kind.name == "USER_INPUT" for spec in program.graph_signature.input_specs))
            _fx._TENSOR_VALUES.update({key: value for key, value in state.items() if value is not None})
            yield
        finally:
            _fx._CHECKER[:] = previous_checker
            for module, name, contents in saved:
                target = getattr(module, name)
                target.clear()
                target.update(contents) if isinstance(target, dict) else target.extend(contents)


def inventory(program, checker):
    """Rule recognition is diagnostic; it is not full conversion or execution."""
    import torch
    from torchlean import _emit, _fx, _graph, _structure
    specs = {spec.arg.name: spec for spec in program.graph_signature.input_specs}
    records, runtime = [], {}
    with capture_state(program, checker):
        helpers = _graph._routing_helpers(program.graph)
        for node in program.graph.nodes:
            spec = specs.get(node.name)
            runtime[node.name] = bool(spec and spec.kind.name == "USER_INPUT") or any(
                runtime[parent.name] for parent in node.all_input_nodes)
            row = {"name": node.name, "op": node.op, "target": str(node.target),
                   "runtime_dependent": runtime[node.name], "result": describe(node.meta.get("val")),
                   "dependencies": [parent.name for parent in node.all_input_nodes],
                   "core_tag": isinstance(node.target, torch._ops.OpOverload) and
                               torch.Tag.core in node.target.tags}
            if node.op in ("placeholder", "get_attr", "output"):
                row["status"] = "declaration"
            elif not node.users and _structure._projection(node) is not None:
                # Serialization reintroduces unused layer-norm statistics. The
                # ordinary importer skips these too; they are not source gaps.
                row["status"] = "ignored_dead_projection"
            else:
                try:
                    if node in helpers or _fx._zeros_like(node):
                        row["kind"] = "routing_helper"
                    elif _graph._metadata_constraint(node) is not None:
                        row["kind"] = "metadata_constraint"
                    elif _structure._component_producer(node):
                        row["kind"] = "producer_lowered_through_projections"
                    else:
                        kind = _emit._lower_kind(node, None)
                        row["kind"] = kind.get("kind", "constant_or_alias")
                    row["status"] = "recognized"
                except Exception as error:
                    row.update(error_record(error))
            records.append(row)
    return records, runtime


def convert(program, graph, input_names, values, checker, *, signature=True):
    import torchlean
    return torchlean._lower_pytorch_graph(None, graph, input_names,
        tensor_values={key: value for key, value in values.items() if key not in input_names},
        signature=program.graph_signature if signature else None, checker=str(checker))[0]


def full_conversion(program, arguments, checker, out):
    values, names = bindings(program, arguments)
    try:
        document = convert(program, program.graph, names, values, checker)
        import torchlean
        (out / "full-model.json").write_text(json.dumps(document, allow_nan=False))
        report = torchlean.verify(document, checker=str(checker), checks=("structure", "types"))
        return {"status": "converted" if report["ok"] else "native_ir_refused", "report": report,
                "native_full_model_execution": "not_run"}
    except Exception as error:
        return error_record(error)


def residual_conversion(program, arguments, records, checker, out, *, extra_cuts=(), filename="residual.json"):
    """Explicit diagnostic boundaries, never a replacement full-model implementation."""
    import torch
    import torchlean
    cuts = {row["name"] for row in records if row["status"] == "unsupported"}
    cuts.update(extra_cuts)
    graph, environment = torch.fx.Graph(), {}
    values, names = bindings(program, arguments)
    for node in program.graph.nodes:
        if node.name in cuts:
            copied = graph.placeholder(node.name)
            names.append(copied.name)
        else:
            copied = graph.node_copy(node, lambda parent: environment[parent])
        copied.meta = dict(node.meta)
        environment[node] = copied
    graph.eliminate_dead_code()
    try:
        document = convert(program, graph, names, values, checker)
        (out / filename).write_text(json.dumps(document, allow_nan=False))
        report = torchlean.verify(document, checker=str(checker), checks=("structure", "types"))
        return {"status": "converted" if report["ok"] else "native_ir_refused",
                "report": report, "cut_count": len(cuts),
                "cuts": sorted(cuts), "nodes_after_dead_code_elimination": len(list(graph.nodes)),
                "scope": "Diagnostic residual only: refused calls are declared runtime inputs. "
                         "Their implementations and full-model execution are NOT checked."}
    except Exception as error:
        return {**error_record(error), "cut_count": len(cuts), "scope": "Diagnostic residual only"}


def native_capsules(program, arguments, records, runtime, checker, out, max_words):
    """Replay real source operations with dynamic cutpoints and frozen ancestors.

    Frozen expressions retain their original operators. Runtime tensors become
    typed input ports, not constants. References come from the untouched PyTorch
    graph. One representative is selected per operator/argument metadata contract.
    """
    import torch
    import torchlean
    from torchlean._fx import _tensor_value
    observed = {}

    class Record(torch.fx.Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            observed[node.name] = value
            return value

    bound, _ = bindings(program, arguments)
    with torch.no_grad():
        Record(program.graph_module).run(*[bound[node.name] for node in program.graph.nodes
                                         if node.op == "placeholder"])
    rows_by_name = {row["name"]: row for row in records}
    seen, results = set(), []
    for target in program.graph.nodes:
        row = rows_by_name[target.name]
        if target.op != "call_function" or row.get("kind") == "metadata_constraint" or \
                row["status"] == "ignored_dead_projection":
            continue
        def contract(item):
            if isinstance(item, torch.fx.Node):
                return describe(item.meta.get("val"))
            if isinstance(item, (tuple, list)):
                return [contract(child) for child in item]
            if isinstance(item, dict):
                return {str(key): contract(child) for key, child in item.items()}
            return describe(item)
        key = json.dumps([str(target.target), contract(target.args), contract(target.kwargs)], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        graph, copied, inputs, frozen = torch.fx.Graph(), {}, [], {}

        def copy_node(node, *, force=False):
            if node in copied:
                return copied[node]
            value = observed[node.name]
            dynamic_tensor = runtime[node.name] and isinstance(value, torch.Tensor)
            if node.op == "placeholder" or dynamic_tensor and not force:
                result = graph.placeholder(node.name)
                if runtime[node.name]:
                    inputs.append((result.name, value))
                else:
                    frozen[result.name] = value
            else:
                result = graph.node_copy(node, lambda parent: copy_node(parent))
            result.meta = dict(node.meta)
            copied[node] = result
            return result

        result = copy_node(target, force=True)
        reference = observed[target.name]
        if isinstance(reference, tuple):
            # The model reads output zero of native_layer_norm. Its unused
            # statistics are not silently demanded of the importer.
            result = graph.call_function(operator.getitem, (result, 0))
            result.meta["val"] = target.meta["val"][0]
            reference = reference[0]
        if not isinstance(reference, torch.Tensor):
            raise TypeError(f"Unexpected capsule result: {target.name}, {type(reference)}")
        graph.output(result)
        words = sum(value.numel() for _, value in inputs) + sum(
            value.numel() for value in frozen.values()) + reference.numel()
        record = {"source_node": target.name, "operator": str(target.target),
                  "argument_contract": contract(target.args), "keywords": contract(target.kwargs),
                  "result": describe(reference), "payload_elements": words}
        if words > max_words:
            record.update(status="not_run", reason="Configured native replay payload bound")
        else:
            try:
                document = convert(program, graph, [name for name, _ in inputs], frozen, checker,
                                   signature=False)
                document["arguments"] = [_tensor_value(value) for _, value in inputs]
                document["reference_outputs"] = [_tensor_value(reference)]
                report = torchlean.verify(document, checker=str(checker), checks=CHECKS)
                record.update(status="compared" if report["ok"] else "native_check_failed", report=report)
                checks = {item["name"]: item["status"] for item in report["checks"]}
                if any(checks.get(name) == "failed" for name in ("structure", "types")):
                    record["status"] = "native_ir_refused"
                if checks.get("execute") == "executed" and checks.get("compare_bitwise") == "failed":
                    record["status"] = "numerical_mismatch"
                    record["numerical_owner"] = str(target.args[0].target) if \
                        target.target is operator.getitem else str(target.target)
                    if reference.dtype == torch.float32:
                        words = report["outputs"][0]["bits"]
                        actual = torch.tensor(words, dtype=torch.int64).to(torch.int32).view(
                            torch.float32).reshape(reference.shape)
                        absolute = (actual.double() - reference.double()).abs()
                        record["max_absolute_error"] = absolute.max().item()
                        record["differing_words"] = int((actual.view(torch.int32) !=
                                                         reference.contiguous().view(torch.int32)).sum())
            except Exception as error:
                record.update(error_record(error))
        results.append(record)
        if len(results) % 10 == 0:
            print(f"Native source capsules: {len(results)}", flush=True)
    (out / "capsules.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    return results


def observed_gaps(records, capsules):
    """Keep source recognition and native IR import as distinct failure stages."""
    gaps = []
    frontend = {row["target"] for row in records if row["status"] == "unsupported"}
    for name in sorted(frontend):
        calls = [row for row in records if row["target"] == name and row["status"] == "unsupported"]
        gaps.append({"operator": name, "stage": "frontend_lowering", "calls": len(calls),
                     "core_tag": calls[0]["core_tag"], "examples": calls[:3]})
    refused = {(row["operator"], row["result"]["dtype"]) for row in capsules
               if row["status"] == "native_ir_refused"}
    for name, dtype in sorted(refused):
        calls = [row for row in records if row["target"] == name and
                 isinstance(row["result"], dict) and row["result"].get("dtype") == dtype]
        examples = [{**{key: value for key, value in row.items() if key != "report"},
                     "checks": row["report"]["checks"]}
                    for row in capsules if row["operator"] == name and row["result"]["dtype"] == dtype
                    and row["status"] == "native_ir_refused"]
        gaps.append({"operator": name, "dtype": dtype, "stage": "native_ir_import",
                     "calls": len(calls), "core_tag": calls[0]["core_tag"],
                     "source_nodes": [row["name"] for row in calls], "examples": examples[:3],
                     "scope": "Source calls requiring the dtype contract refused by representative native capsules"})
    return gaps


def integer_core_probe(checker):
    """Does integer dot-product arithmetic expressed with Core operators import?"""
    import torch
    import torchlean
    from torchlean._fx import _tensor_value

    class Product(torch.nn.Module):
        def forward(self, left, right):
            return torch._int_mm(left, right)

    left = torch.arange(-32, 32, dtype=torch.int8).reshape(8, 8)
    right = left.t().contiguous()
    with torch.no_grad():
        core = torch.export.export(Product(), (left, right), strict=False).run_decompositions(core_decompositions())
        expected = torch._int_mm(left, right)
        actual = core.module()(left, right)
        assert actual.dtype == torch.int32 and torch.equal(actual, expected)
    values, names = bindings(core, (left, right))
    document = convert(core, core.graph, names, values, checker)
    document["arguments"] = [_tensor_value(left), _tensor_value(right)]
    document["reference_outputs"] = [_tensor_value(expected)]
    report = torchlean.verify(document, checker=str(checker), checks=CHECKS)
    return {"purpose": "Small control using the same widening-to-INT32 decomposition as the main model",
            "pytorch_matches_int_mm": True,
            "all_aten_calls_core_tagged": all(torch.Tag.core in node.target.tags for node in core.graph.nodes
                if node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload)),
            "all_numerical_calls_core_tagged": all(torch.Tag.core in node.target.tags for node in core.graph.nodes
                if node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload)
                and str(node.target) != "aten._assert_tensor_metadata.default"),
            "operators": [str(node.target) for node in core.graph.nodes if node.op == "call_function"],
            "checks": report["checks"], "ok": report["ok"], "scope": "Small signed INT8 input fixture"}


def render(report):
    def escape(value):
        return html.escape(str(value))
    groups = report["operators"]
    gaps = report["gaps"]
    rows = "".join(f"<tr><td><code>{escape(row['operator'])}</code></td><td>{row['calls']}</td>"
                   f"<td>{'Core' if row['core_tag'] else 'Additional'}</td>"
                   f"<td>{escape(row['statuses'])}</td></tr>" for row in groups)
    gap_cards = "".join(f"<article><h3>{escape(row['operator'])}</h3>"
        f"<p>{row['calls']} source calls require this unsupported contract. "
        f"Stage: <strong>{escape(row['stage'])}</strong>. Dtype: {escape(row.get('dtype', 'see example'))}. "
        f"{'Core ATen operator.' if row['core_tag'] else 'Non-Core operator.'}</p>"
        f"<pre>{escape(json.dumps(row['examples'], indent=2))}</pre></article>" for row in gaps)
    parity = "".join(f"<tr><td>{escape(row['case'])}</td><td>{row['raw_bitwise_equal']}</td>"
                      f"<td>{row['core_bitwise_equal']}</td><td>{row['roundtrip_bitwise_equal']}</td></tr>"
                      for row in report["parity"])
    native = report["native_capsule_summary"]
    differences = "".join(f"<tr><td><code>{escape(row['source_node'])}</code></td>"
        f"<td><code>{escape(row['numerical_owner'])}</code></td>"
        f"<td>{row.get('differing_words', '?')}</td><td>{row.get('max_absolute_error', '?')}</td></tr>"
        for row in report["native_capsules"] if row["status"] == "numerical_mismatch")
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>TinyStories: gaps after Core ATen decomposition</title>
<style>body{{font:16px/1.55 system-ui;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#172033}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}td,th{{border:1px solid #ccd4df;padding:.5rem;text-align:left}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f9;padding:1rem}}article{{border-left:4px solid #bd4e15;padding-left:1rem}}
code{{font-family:ui-monospace,monospace}}.note{{background:#fff4d8;padding:1rem}}small{{color:#526072}}</style>
<h1>TinyStories-1M: gaps after Core ATen decomposition</h1>
<p>Measured at <code>{escape(report['provenance']['git_revision'])}</code>.
CPU TorchAO W8A8; explicit runtime token IDs and attention mask, both [1,32].</p>
<pre>quantize_(model, Int8DynamicActivationInt8WeightConfig(version=2))
ep = torch.export.export(model, (input_ids, attention_mask), strict=False)
def int_mm_to_core(left, right):
    return torch.ops.aten.mm.default(left.to(torch.int32), right.to(torch.int32))
table = torch.export.default_decompositions()
table[torch.ops.aten._int_mm.default] = int_mm_to_core
core = ep.run_decompositions(table)</pre>
<p>PyTorch documents this <a href="https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export.html#export-ir-training-vs-inference">export/decomposition workflow</a>.
TorchLean's importer accepts default export and Core ATen forms; no TorchLean runtime code was changed.</p>
<h2>Observed frontend and native IR gaps</h2>{gap_cards}
<p>Decomposition removes the raw graph's TorchAO calibration/quantization calls, tensor-subclass accessors,
in-place additions and in-place detaches. Their source counts before/after decomposition are in the JSON report.
Embedding remains present and is recognized; the export path does not have the live-weight compiler refusal.</p>
<p class="note">Whole-model import: <strong>{escape(report['whole_model']['status'])}</strong>.
{escape(report['whole_model'].get('message', ''))}.
The native INT32 matrix-product refusal appears after isolating the frontend indexing blocker.
Local rule recognition alone does not establish native IR acceptance or numerical equivalence.</p>
<h2>Is this exactly the “~180 operators”?</h2>
<p>The graph has {report['core_tagged_operator_count']} distinct Core-tagged operators.
All numerical ATen calls are Core-tagged. The non-Core scaffolding consists of metadata assertions and tuple projections.
The raw graph's 49 <code>aten._int_mm.default</code> calls become 49 Core <code>aten.mm.default</code> calls.
Each product explicitly widens both INT8 operands to INT32 first; merely renaming the INT8 operation would change its result dtype.</p>
<p>A small control uses the same decomposition and agrees with <code>_int_mm</code> in PyTorch.
TorchLean refuses the INT32 matrix-product IR record. Supporting an operator name alone does not establish
support for every dtype or accumulator contract.</p>
<h2>Quantized behavior preserved in PyTorch</h2>
<p>All 49 linear layers use INT8 weights and dynamic per-token INT8 activations.
The decomposed graph expresses those 49 products as INT8→INT32 casts followed by INT32 matrix multiplication.
This reference representation does not prescribe 32-bit storage or multipliers for an eventual FPGA implementation.</p>
<table><tr><th>Input case</th><th>Raw export vs eager</th><th>Decomposed vs eager</th><th>Saved/reloaded core vs eager</th></tr>{parity}</table>
<p>Equality above compares every output byte. This is sampled PyTorch equivalence, not native TorchLean full-model execution.</p>
<h2>Conversion and native-check evidence</h2>
<p>After isolating only the frontend refusal: <strong>{escape(report['residual_frontend_only']['status'])}</strong>.
This diagnostic still contains the INT32 matrix products and exposes their native IR import refusal.</p>
<p>Diagnostic residual conversion: <strong>{escape(report['residual']['status'])}</strong>.
{escape(report['residual'].get('scope', ''))}</p>
<p>Native source capsules: {escape(native)}. Runtime cutpoints use real PyTorch intermediate values;
frozen ancestor expressions retain their source operations. Payloads above {report['native_max_words']} elements are explicitly not run.</p>
<h3>Native numerical mismatches</h3>
<table><tr><th>Source case</th><th>Numerical operator</th><th>Different output words</th><th>Maximum absolute error</th></tr>{differences}</table>
<p>Layer norm and its dependent <code>getitem</code> repeat the same mismatch; they are one numerical family.
Two <code>bmm</code> shapes also belong to one family. These sampled differences need investigation or an explicit numerical
acceptance policy. This report does not attribute every difference to an allowed PyTorch approximation.</p>
<p class="note">Fixing the {len(gaps)} observed import gaps is not sufficient evidence of full-model bitwise correctness.
The native comparisons above already expose numerical differences. Native full-model execution remains blocked.</p>
<p>Untested cases and complete native check statuses are recorded in the JSON report.
Metadata assertions are capture constraints and are not numerical kernels.</p>
<h2>Complete source operator inventory</h2>
<p>Raw export: {report['graph']['raw_nodes']} nodes. Decomposed: {report['graph']['core_nodes']} nodes.
Saved/reloaded: {report['graph']['roundtrip_nodes']} nodes; serialization adds unused layer-norm statistic projections.
Dead projections are ignored by the ordinary importer and are not reported as gaps.</p>
<table><tr><th>Operator</th><th>Calls</th><th>Boundary</th><th>Local rule status</th></tr>{rows}</table>
<h2>Reproduce</h2><pre>{escape(report['reproduction'])}</pre>
<p>Full graph text, exported programs, node inventory, native capsule records and residual artifact are in
<code>{escape(report['artifact_directory'])}</code>. The adjacent JSON report records hashes and package versions.</p>
<h2>Provenance</h2><pre>{escape(json.dumps(report['provenance'], indent=2))}</pre>
</html>
"""


def run(args):
    import torch
    import torchao
    import torchlean
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
    from transformers import GPTNeoForCausalLM
    if os.environ.get("TORCHLEAN_OP_TABLE"):
        raise ValueError("Unset TORCHLEAN_OP_TABLE for a measured runtime survey")
    sources = source_hashes()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("Loading cached TinyStories-1M", flush=True)
    inner = GPTNeoForCausalLM.from_pretrained(args.snapshot, local_files_only=True,
        attn_implementation="eager").float().cpu().eval()
    quantize_(inner, Int8DynamicActivationInt8WeightConfig(version=2))
    linear = [module for module in inner.modules() if isinstance(module, torch.nn.Linear)]
    assert len(linear) == 49 and all(module.weight.qdata.dtype == torch.int8 for module in linear)

    class Logits(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.inner = model

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask,
                              use_cache=False, return_dict=False)[0]

    model = Logits(inner).eval()
    ids = torch.tensor([TOKENS], dtype=torch.int64)
    mask = torch.ones_like(ids)
    padded_ids, padded_mask = ids.clone(), mask.clone()
    padded_ids[:, -8:] = 0
    padded_mask[:, -8:] = 0
    cases = [("dense", (ids, mask)), ("padded_tail", (padded_ids, padded_mask)),
             ("different_tokens", (ids.flip(-1), mask))]
    with torch.no_grad():
        expected = [model(*arguments) for _, arguments in cases]
        print("torch.export.export", flush=True)
        raw = torch.export.export(model, (ids, mask), strict=False)
        print("run_decompositions(default table + _int_mm widening)", flush=True)
        core = raw.run_decompositions(core_decompositions())
        raw_count, core_count = len(list(raw.graph.nodes)), len(list(core.graph.nodes))
        (out / "raw.graph.txt").write_text(str(raw.graph))
        (out / "core.graph.txt").write_text(str(core.graph))
        torch.export.save(raw, out / "raw.pt2")
        torch.export.save(core, out / "core.pt2")
        restored = torch.export.load(out / "core.pt2")
        parity = []
        for (name, arguments), reference in zip(cases, expected):
            def equal(value):
                return value.dtype == reference.dtype and value.shape == reference.shape and torch.equal(
                    value.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8))
            row = {"case": name, "raw_bitwise_equal": equal(raw.module()(*arguments)),
                   "core_bitwise_equal": equal(core.module()(*arguments)),
                   "roundtrip_bitwise_equal": equal(restored.module()(*arguments)),
                   "finite": bool(torch.isfinite(reference).all()), "output": describe(reference)}
            assert all(row[key] for key in ("raw_bitwise_equal", "core_bitwise_equal", "roundtrip_bitwise_equal")), row
            parity.append(row)
            print("Bitwise PyTorch parity:", name, flush=True)
    # The serialized graph is the reproducible primary analysis input. Its extra
    # dead tuple projections are recorded explicitly rather than counted as gaps.
    records, runtime = inventory(restored, args.checker)
    assert not any(row["status"] == "error" for row in records), "Unexpected diagnostic error"
    groups = {}
    for row in records:
        if row["status"] == "declaration":
            continue
        group = groups.setdefault(row["target"], {"operator": row["target"], "calls": 0,
            "core_tag": row["core_tag"], "statuses": Counter(), "examples": []})
        group["calls"] += 1
        group["statuses"][row["status"]] += 1
        if row["status"] == "unsupported" and len(group["examples"]) < 3:
            group["examples"].append(row)
    (out / "inventory.json").write_text(json.dumps(records, indent=2, allow_nan=False))
    print("Checking original decomposed graph", flush=True)
    whole = full_conversion(restored, (ids, mask), args.checker, out)
    print("Checking diagnostic residual after frontend refusals", flush=True)
    residual_frontend = residual_conversion(restored, (ids, mask), records, args.checker, out,
                                            filename="residual-frontend-only.json")
    print("Native source-operation replay", flush=True)
    with torch.no_grad():
        capsules = native_capsules(restored, (ids, mask), records, runtime,
                                   args.checker, out, args.max_words)
    integer_probe = integer_core_probe(args.checker)
    gaps = observed_gaps(records, capsules)
    native_cuts = {name for gap in gaps if gap["stage"] == "native_ir_import" for name in gap["source_nodes"]}
    print("Checking diagnostic residual after all observed import gaps", flush=True)
    residual = residual_conversion(restored, (ids, mask), records, args.checker, out, extra_cuts=native_cuts)
    metadata = Counter(row["status"] for row in records)
    assert sum(metadata.values()) == len(list(restored.graph.nodes))
    counts = Counter(str(node.target) for node in restored.graph.nodes if node.op == "call_function")
    assert counts["aten._int_mm.default"] == 0
    assert counts["aten.mm.default"] == 49
    assert not any(key.startswith("torchao.") for key in counts)
    non_core_numerical = sorted({str(node.target) for node in restored.graph.nodes
        if node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload)
        and torch.Tag.core not in node.target.tags
        and node.target != torch.ops.aten._assert_tensor_metadata.default})
    assert not non_core_numerical, non_core_numerical
    raw_integer = [node for node in raw.graph.nodes if str(node.target) == "aten._int_mm.default"]
    assert len(raw_integer) == 49
    assert all(parent.meta["val"].dtype == torch.int8 for node in raw_integer for parent in node.all_input_nodes)
    for node in restored.graph.nodes:
        if str(node.target) == "aten.mm.default":
            assert node.meta["val"].dtype == torch.int32
            assert all(parent.meta["val"].dtype == torch.int32 for parent in node.all_input_nodes)
    assert source_hashes() == sources, "TorchLean source changed while this survey was running; rerun it"
    # Full native output words remain in the local capsules.json evidence, not
    # duplicated into the smaller, reviewable documentation report.
    compact_capsules = []
    for capsule in capsules:
        compact = {key: value for key, value in capsule.items() if key != "report"}
        if "report" in capsule:
            compact["checks"] = capsule["report"]["checks"]
        compact_capsules.append(compact)
    report = {"format": "torchlean.tinystories-core-aten-gaps.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "torch.export.export(Logits(quantized_model), (input_ids, attention_mask), strict=False).run_decompositions(default_decompositions + _int_mm_to_int32_mm)",
        "normalization": {"extra_rule": "aten._int_mm.default -> INT32 casts + aten.mm.default",
                          "raw_int8_products": len(raw_integer), "remaining_int_mm_calls": counts["aten._int_mm.default"],
                          "int32_core_mm_calls": counts["aten.mm.default"],
                          "non_core_numerical_operators": non_core_numerical},
        "model": {"name": "roneneldan/TinyStories-1M", "snapshot": str(args.snapshot.resolve()),
            "files": {path.name: digest(path) for path in args.snapshot.iterdir()
                      if path.name in ("config.json", "pytorch_model.bin", "model.safetensors")},
            "quantization": "Int8DynamicActivationInt8WeightConfig(version=2)",
            "quantized_linear_layers": len(linear), "kv_cache": False,
            "input_ids": TOKENS, "attention_mask": [1] * 32, "input_shape": [1, 32]},
        "parity": parity, "whole_model": whole, "residual_frontend_only": residual_frontend,
        "residual": residual, "gaps": gaps, "integer_core_probe": integer_probe,
        "removed_refusals": {name: {"raw_calls": sum(str(node.target) == name for node in raw.graph.nodes),
                                    "core_calls": counts[name]}
            for name in ("torchao.choose_qparams_affine.default", "torchao.quantize_affine.default",
                         "export.access_subclass_inner_tensor.default", "aten.add_.Tensor", "aten.detach_.default",
                         "aten._int_mm.default")},
        "graph": {"raw_nodes": raw_count, "core_nodes": core_count,
            "roundtrip_nodes": len(records), "raw_graph_sha256": digest(out / "raw.graph.txt"),
            "core_graph_sha256": digest(out / "core.graph.txt"),
            "raw_archive_sha256": digest(out / "raw.pt2"), "core_archive_sha256": digest(out / "core.pt2")},
        "coverage": dict(metadata), "operators": sorted(groups.values(), key=lambda row: row["operator"]),
        "core_tagged_operator_count": sum(row["core_tag"] for row in groups.values()),
        "native_max_words": args.max_words,
        "native_capsule_summary": dict(Counter(row["status"] for row in capsules)),
        "native_capsules": compact_capsules, "artifact_directory": os.path.relpath(out, ROOT),
        "provenance": {"git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchao", "transformers")},
            "checker_sha256": digest(args.checker), "python_frontend_sha256": sources,
            "diagnostic_sha256": digest(Path(__file__))},
        "reproduction": f".venv/bin/python scripts/checks/tinystories_core_aten_gaps.py --snapshot {args.snapshot} --out {args.out} --report {args.report}"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.report.with_suffix(".html").write_text(render(report))
    print(json.dumps({"whole_model": whole, "residual_status": residual["status"],
        "gaps": [{key: value for key, value in gap.items() if key not in ("examples", "source_nodes")}
                 for gap in gaps],
        "native_capsules": report["native_capsule_summary"], "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("_out/tinystories-core-aten"))
    parser.add_argument("--report", type=Path, default=Path("docs/TINYSTORIES_CORE_ATEN_GAPS"))
    parser.add_argument("--checker", type=Path, default=ROOT / ".lake/build/bin/verify")
    parser.add_argument("--max-words", type=int, default=100000)
    run(parser.parse_args())
