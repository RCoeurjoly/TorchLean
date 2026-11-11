"""Measure the current TorchLean gaps of a pinned TorchAO W8A8 TinyStories graph."""

from __future__ import annotations

import argparse
import copy
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

from tinystories_torchao_inventory import CurrentAdapter, capture_state, inspect_export
from render_tinystories_torchao_survey import render_html, rank_gaps

ROOT = Path(__file__).resolve().parents[2]
BASE = 'c2ff005b2fb2a36ca0f09c6218c7eeb9fd667751'
REVISION = 'ac533fb8b4f69c71894bf96badfe11e6294d9fcf'
CHECKPOINT_HASHES = {
    'pytorch_model.bin': '07f9609ea882b8163ff3b23d40e2b82cb715d409631beb15c84b164f3877dae7',
    'config.json': 'ff74c30d5ebb5ab1da0f2ea479adf7197c504b42b5522a858c334ab91ed4958c',
}
TOKENS = [7454, 2402, 257, 640, 11, 257, 1310, 2576, 3706, 20037, 1043, 257,
          6016, 2266, 2613, 287, 262, 11376, 13, 198, 3347, 3751, 340, 284, 607,
          3956, 11, 290, 1978, 484, 11686, 340]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_checkpoint(snapshot):
    actual = {name: digest(Path(snapshot)/name) for name in CHECKPOINT_HASHES}
    if actual != CHECKPOINT_HASHES:
        raise ValueError('Checkpoint/config differs from the pinned TinyStories-1M revision: ' + str(actual))
    return actual


def inspect_program(program, checker):
    import torch
    before = str(program.graph)
    with capture_state(program, checker) as adapter:
        source = inspect_export(program, adapter)
    assert before == str(program.graph), 'Inventory mutated the source graph'
    assert len(source['nodes']) == len(list(program.graph.nodes))
    source['node_count'] = len(source['nodes'])
    originals = {node.name: node for node in program.graph.nodes}
    for record in source['nodes']:
        target = originals[record['id']].target
        if not isinstance(target, torch._ops.OpOverload) and callable(target):
            module, name = getattr(target, '__module__', None), getattr(target, '__qualname__', None)
            if module and name:
                record['target'] = module + '.' + name
    source['scope'] = 'Local rule/emitter recognition only; no full conversion or execution implied'
    return source


def summarize(source):
    groups = {}
    for node in source['nodes']:
        if node['op'] in ('placeholder', 'get_attr', 'output'):
            continue
        row = groups.setdefault(node['target'], dict(operator=node['target'], calls=0,
            recognized=0, unsupported=0, probe_errors=0, unchecked=0, runtime_calls=0, examples=[]))
        row['calls'] += 1
        row['runtime_calls'] += int(node.get('depends_on_runtime_input', False))
        status = node.get('probe', {}).get('status')
        row[{'recognized': 'recognized', 'unsupported': 'unsupported',
             'probe_error': 'probe_errors'}.get(status, 'unchecked')] += 1
        if status != 'recognized' and len(row['examples']) < 4:
            row['examples'].append(node)
    return sorted(groups.values(), key=lambda row: row['operator'])


def failure(error):
    pending, seen, unsupported = [error], set(), False
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        unsupported |= isinstance(item, NotImplementedError) or (
            type(item).__module__.startswith('torch._dynamo') and type(item).__name__ == 'Unsupported')
        pending.extend(nested for nested in (item.__cause__, item.__context__,
                       getattr(item, 'inner_exception', None)) if isinstance(nested, BaseException))
    result = {'status': 'refused' if unsupported else 'error',
            'error_type': type(error).__name__, 'message': str(error),
            'traceback': traceback.format_exc()}
    trace = error.__traceback__
    while trace:
        node = trace.tb_frame.f_locals.get('node')
        if hasattr(node, 'name') and hasattr(node, 'target'):
            result['source_node'] = {'id': node.name, 'operator': str(node.target)}
        trace = trace.tb_next
    return result


def convert_export(program, checker, out):
    import torchlean
    values = {}
    state, constants = dict(program.state_dict), dict(program.constants)
    inputs = []
    for spec in program.graph_signature.input_specs:
        if spec.kind.name == 'USER_INPUT':
            inputs.append(spec.arg.name)
        else:
            values[spec.arg.name] = state.get(spec.target, constants.get(spec.target))
    try:
        document, _ = torchlean._lower_pytorch_graph(None, program.graph, inputs,
            signature=program.graph_signature, tensor_values=values, checker=str(checker))
        (out / 'full-model.json').write_text(json.dumps(document, allow_nan=False))
        result = torchlean.verify(document, checker=str(checker), checks=('structure', 'types'))
        return {'status': 'converted', 'native': result,
                'full_model_execution': 'not_run', 'full_model_bitwise_comparison': 'not_run'}
    except Exception as error:
        return failure(error)


def compile_path(model, tokens, checker):
    import torch
    import torchlean
    wrapped = torchlean.compile(type(model)(model.inner), checker=str(checker))
    try:
        with torch.no_grad():
            wrapped(tokens)
        result = {'status': 'checked_and_executed', 'native': wrapped.torchlean.report}
    except Exception as error:
        result = failure(error)

    # Inspect the graph at the ordinary backend's lowering boundary. This run is
    # deliberately aborted after collecting evidence; it never substitutes an unchecked graph.
    class InventoryComplete(Exception):
        pass

    captured = {}
    from torchlean import _pytorch
    ordinary_lower = _pytorch.lower

    def track(graph_module, inputs, checker=None):
        originals = [node for node in graph_module.graph.nodes if node.op == 'placeholder']
        bindings = dict(zip(originals, inputs))
        wrappers = []
        for node, value in bindings.items():
            if hasattr(value, 'qdata') and hasattr(value, 'scale'):
                wrappers.append({'input': node.name, 'class': type(value).__module__ + '.' + type(value).__qualname__,
                    'logical_dtype': str(value.dtype), 'shape': list(value.shape),
                    'storage_dtype': str(value.qdata.dtype), 'scale_dtype': str(value.scale.dtype)})
        linear = [node for node in graph_module.graph.nodes
                  if node.op == 'call_function' and node.target is torch.nn.functional.linear]
        captured['live_quantized_inputs'] = wrappers
        captured['original_linear_calls'] = len(linear)
        captured['quantized_linear_calls'] = sum(node.args[1] in bindings and hasattr(bindings[node.args[1]], 'qdata')
                                                for node in linear)
        captured['before_backend_translation_nodes'] = len(list(graph_module.graph.nodes))
        return ordinary_lower(graph_module, inputs, checker)

    def collect(gm, graph, input_names, **kwargs):
        specs = [SimpleNamespace(arg=SimpleNamespace(name=n.name),
                 kind=SimpleNamespace(name='USER_INPUT'), target=None)
                 for n in graph.nodes if n.op == 'placeholder']
        program = SimpleNamespace(graph=graph,
            graph_signature=SimpleNamespace(input_specs=specs, output_specs=[]),
            state_dict={}, constants={})
        captured.update(inspect_program(program, checker))
        raise InventoryComplete('Stopped after read-only frontend inventory')

    try:
        with patch('torchlean._lower_pytorch_graph', collect), patch('torchlean._pytorch.lower', track):
            diagnostic = torchlean.compile(type(model)(model.inner), checker=str(checker))
            with torch.no_grad():
                diagnostic(tokens)
    except Exception as error:
        if not captured.get('nodes'):
            result['inventory_error'] = failure(error)
    if captured.get('live_quantized_inputs'):
        result['quantized_representation'] = {
            'status': 'representation_gap', 'quantized_linear_calls': captured.get('quantized_linear_calls'),
            'message': 'Live Int8Tensor weights have logical float32 dtype. The ordinary backend rewrites '
                       'their functional linear calls to transpose/matmul nodes without explicitly lifting '
                       'INT8 payloads, per-token activation quantization and scales into the checked IR. '
                       'Local recognition of those floating forms is not recognition of the W8A8 computation.'}
    return result, captured


def native_fixtures(checker):
    import torch
    import torchlean

    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lookup = torch.nn.Embedding(8, 4)

        def forward(self, tokens):
            return self.lookup(tokens)

    class IntegerMM(torch.nn.Module):
        def forward(self, a, b):
            return torch._int_mm(a, b)

    class IntegerClamp(torch.nn.Module):
        def forward(self, x):
            return x.clamp(-128, 127)

    class Cast(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.float32)

    class Amax(torch.nn.Module):
        def forward(self, x):
            return x.abs().amax(dim=-1, keepdim=True)

    class Round(torch.nn.Module):
        def forward(self, x):
            return x.round()

    class ChooseQParams(torch.nn.Module):
        def forward(self, x):
            return torch.ops.torchao.choose_qparams_affine.default(x, 'SYMMETRIC', [1, 1, 4],
                torch.int8, -128, 127, torch.finfo(torch.float32).eps, torch.float32, torch.int8, True)

    class QuantizeAffine(torch.nn.Module):
        def forward(self, x, scale, zero):
            return torch.ops.torchao.quantize_affine.default(x, [1, 1, 4], scale, zero, torch.int8, -128, 127)

    class AddInplace(torch.nn.Module):
        def forward(self, x, bias):
            y = x + 1.
            return y.add_(bias)

    class DetachInplace(torch.nn.Module):
        def forward(self, x):
            value = torch.tensor(-0.)
            value.detach_()
            return x + value

    cases = [('embedding', Embedding().eval(), (torch.tensor([[0, 3, 7]]),)),
             ('int8_matmul', IntegerMM(), (torch.arange(-32, 32, dtype=torch.int8).reshape(8, 8),
                                          torch.arange(-32, 32, dtype=torch.int8).reshape(8, 8))),
             ('int32_clamp', IntegerClamp(), (torch.tensor([-200, -128, -1, 0, 127, 200], dtype=torch.int32),)),
             ('int32_to_float32', Cast(), (torch.tensor([-16777217, -1, 0, 1, 16777217], dtype=torch.int32),)),
             ('int8_to_float32', Cast(), (torch.tensor([-128, -1, 0, 1, 127], dtype=torch.int8),)),
             ('per_token_amax', Amax(), (torch.tensor([[-2., 1., 0.], [-0., 0., 0.], [.5, -4., 3.]]),)),
             ('round_ties_even', Round(), (torch.tensor([-2.5, -1.5, -.5, .5, 1.5, 2.5]),)),
             ('torchao_choose_qparams', ChooseQParams(), (torch.tensor([[[-2., -1., 0., 1.], [0., 0., 0., 0.]]]),)),
             ('torchao_quantize_affine', QuantizeAffine(),
                 (torch.tensor([[[-2., -1., 0., 1.], [-.05, .05, .15, .25]]]),
                  torch.full((1, 2, 1), .1), torch.zeros(1, 2, 1, dtype=torch.int8))),
             ('fresh_add_inplace', AddInplace(), (torch.ones(2, 4), torch.arange(4, dtype=torch.float32))),
             ('frozen_detach_inplace', DetachInplace(), (torch.tensor([-0., 0., 1.]),))]
    results = []
    for name, model, args in cases:
        expected = None
        from torchlean._fx import _tensor_bits, _dtype_of
        def words(value):
            if isinstance(value, torch.Tensor):
                return {'shape': list(value.shape), 'dtype': _dtype_of(value), 'bits': _tensor_bits(value)}
            return [words(item) for item in value]
        try:
            with torch.no_grad():
                expected = words(model(*args))
            result = torchlean.verify(model, *args, checker=str(checker),
                                      checks=('structure', 'types', 'execute', 'compare_bitwise'))
            results.append({'name': name, 'status': 'bitwise_checked' if result['ok'] else 'refused',
                            'report': result, 'inputs': [words(arg) for arg in args],
                            'expected_eager_words': expected,
                            'scope': 'Small CPU fixture only; not a full-model proof'})
        except Exception as error:
            results.append({'name': name, **failure(error), 'scope': 'Small CPU fixture only',
                            'expected_eager_words': expected})
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
    linear = torch.nn.Sequential(torch.nn.Linear(8, 8)).eval()
    quantize_(linear, Int8DynamicActivationInt8WeightConfig(version=2))
    logical, storage = str(linear[0].weight.dtype), str(linear[0].weight.qdata.dtype)
    compiled = torchlean.compile(linear, checker=str(checker))
    try:
        with torch.no_grad():
            compiled(torch.randn(2, 8))
        document = compiled.torchlean.artifact
        tags = [node['op']['record']['tag'] for node in document['artifact']['nodes'] if 'op' in node]
        results.append({'name': 'quantized_linear_public_compile', 'status': 'structure_checked_only',
            'scope': 'Typing success is not quantized frontend equivalence',
            'logical_weight_dtype': logical, 'stored_weight_dtype': storage,
            'native': compiled.torchlean.report, 'artifact': document, 'ir_operation_tags': tags})
    except Exception as error:
        results.append({'name': 'quantized_linear_public_compile', **failure(error)})
    return results


def run(args):
    import torch
    import torchao
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
    from transformers import GPTNeoForCausalLM
    if os.environ.get('TORCHLEAN_OP_TABLE'):
        raise ValueError('External operator-table overrides are forbidden in a measured survey')
    changed = subprocess.check_output(['git', 'diff', BASE, '--', 'python/torchlean',
        'TorchLean', 'csrc', 'lakefile.lean', 'lake-manifest.json', 'lean-toolchain'], cwd=ROOT)
    if changed:
        raise ValueError('Runtime or dependency sources differ from the stated baseline')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Use a new or empty output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    os.environ['TORCHLEAN_VERIFY'] = str(args.checker)
    from torchlean._table import op_table
    table = op_table(str(args.checker))
    (args.out / 'op-table.json').write_text(json.dumps(table, indent=2))
    snapshot = Path(args.snapshot)
    if not snapshot.is_dir():
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(args.snapshot, revision=REVISION,
                        allow_patterns=['config.json', 'pytorch_model.bin']))
    checkpoint_hashes = validate_checkpoint(snapshot)
    inner = GPTNeoForCausalLM.from_pretrained(snapshot, local_files_only=True,
                        attn_implementation='eager').float().cpu().eval()

    class Logits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, tokens):
            return self.inner(tokens, use_cache=False, return_dict=False)[0]

    tokens = torch.tensor([TOKENS], dtype=torch.int64)
    quantize_(inner, Int8DynamicActivationInt8WeightConfig(version=2))
    linear = [module for module in inner.modules() if isinstance(module, torch.nn.Linear)]
    assert all(hasattr(module.weight, 'qdata') and module.weight.qdata.dtype == torch.int8 for module in linear)
    model = Logits(inner).eval()
    with torch.no_grad():
        expected = model(tokens)
        program = torch.export.export(model, (tokens,), strict=False)
        actual = program.module()(tokens)
        assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    torch.export.save(program, args.out / 'model.pt2')
    program = torch.export.load(args.out / 'model.pt2')
    with torch.no_grad():
        restored = program.module()(tokens)
        assert torch.equal(restored.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    source = inspect_program(program, args.checker)
    exported = convert_export(program, args.checker, args.out)
    compiled, dynamo = compile_path(model, tokens, args.checker)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in (ROOT/'python/torchlean').glob('*.py')}
    checkpoint = snapshot/'pytorch_model.bin'
    report = {'format': 'torchlean.torchao-tinystories-gaps.v1',
        'title': 'TinyStories-1M: TorchAO W8A8 gaps', 'runtime_revision': BASE,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'model': {'checkpoint_revision': REVISION, 'snapshot': str(snapshot),
                  'checkpoint_sha256': checkpoint_hashes['pytorch_model.bin'],
                  'checkpoint_file_hashes': checkpoint_hashes,
                  'export_sha256': digest(args.out/'model.pt2'),
                  'graph_sha256': hashlib.sha256(str(program.graph).encode()).hexdigest(),
                  'quantization': 'Int8DynamicActivationInt8WeightConfig(version=2)',
                  'weights': 'symmetric int8 per output channel',
                  'activations': 'dynamic symmetric int8 per token',
                  'input': {'dtype': 'int64', 'shape': [1, 32], 'values': TOKENS},
                  'output_shape': list(actual.shape), 'eager_export_bitwise_equal': True,
                  'export_roundtrip_bitwise_equal': True,
                  'finite_logits': bool(torch.isfinite(actual).all()), 'kv_cache': False},
        'provenance': {'analysis_revision': revision, 'runtime_matches_baseline': True,
                       'python': platform.python_version(),
                       'packages': {name: importlib.metadata.version(name) for name in ('torch', 'torchao', 'transformers')},
                       'checker_path': str(args.checker), 'checker_sha256': digest(args.checker),
                       'op_table_sha256': digest(args.out/'op-table.json'),
                       'runtime_source_hashes': source_hashes,
                       'dependency_source_hashes': {name: digest(ROOT/name) for name in ('lake-manifest.json', 'lean-toolchain', 'lakefile.lean')},
                       'diagnostic_source_hashes': {p.name: digest(p) for p in Path(__file__).parent.glob('*tinystories_torchao*.py')}},
        'paths': {'export': 'torch.export.export(Logits(model), (tokens,), strict=False)',
                  'compile': 'torchlean.compile(Logits(model))(tokens)'},
        'whole_model': {'export': exported, 'compile': compiled},
        'inventories': {'export': source, 'compile': dynamo}, 'fixtures': native_fixtures(args.checker),
        'warnings': getattr(args, '_stderr', io.StringIO()).getvalue().splitlines(),
        'reproduction': 'See docs/TINYSTORIES_TORCHAO_SURVEY.md for the pinned environment and command.'}
    report['features'] = {name: summarize(inventory) for name, inventory in report['inventories'].items() if inventory}
    report['model']['quantized_linear_layers'] = len(linear)
    report['coverage_complete'] = all(inventory.get('nodes') for inventory in report['inventories'].values())
    report['gaps'] = rank_gaps(report['features'])
    if compiled.get('quantized_representation'):
        report['gaps'].append({'id': 'F00', 'operator': 'Live quantized parameter / linear representation',
            'category': 'compile', 'difficulty': 'Hard',
            'calls': compiled['quantized_representation']['quantized_linear_calls'],
            'status': 'representation_gap', 'probe_errors': 0,
            'guidance': compiled['quantized_representation']['message'] + ' Normalize the tensor subclass '
                        'to explicit typed payload/scale inputs and preserve the activation quantizer and '
                        'integer dot-product/rescale order. Reuse the exported ATen contract as the reference; '
                        'do not replace the actual W8A8 program with floating GEMM.',
            'examples': captured_inputs(dynamo)})
    path = args.out/'report.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.out/'report.html').write_text(render_html(report))
    print(json.dumps({'report': str(path), 'gaps': len(report['gaps']),
                      'nodes': {k:v.get('node_count') for k,v in report['inventories'].items()},
                      'whole_model': {k:v['status'] for k,v in report['whole_model'].items()}}, indent=2))


def captured_inputs(source):
    return source.get('live_quantized_inputs', [])[:4]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', default='roneneldan/TinyStories-1M')
    parser.add_argument('--checker', type=Path, default=ROOT/'.lake/build/bin/verify')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args._stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(args._stderr):
            run(args)
    finally:
        sys.stderr.write(args._stderr.getvalue())
