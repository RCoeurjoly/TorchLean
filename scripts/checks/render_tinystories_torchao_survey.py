"""Standalone HTML presentation of measured TorchAO TinyStories gap evidence."""

import argparse
import html
import json
from pathlib import Path


NOTES = {
    'torch.nn.functional.layer_norm': ('Hard',
        'The existing normalization importer expects resolvable affine parameter payloads. '
        'The compiler path instead supplies live gamma/beta tensor inputs, so simply recognizing '
        'the function name is insufficient. Preserve those inputs in the IR or use an explicitly '
        'frozen export contract; do not silently freeze parameters of a live compiled module. '
        'Reuse current normalization arithmetic and FloatLib codecs where their rounding agrees.'),
    'torch.nn.functional.embedding': ('Small',
        'Normalize the observed functional argument order and defaults to the existing ATen '
        'embedding/gather rule, retaining live weight inputs and signed index checks. The exported '
        'ATen form and a native bitwise fixture already work. This is a frontend representation '
        'gap, not evidence that embedding arithmetic is absent.'),
    '_operator.getitem': ('Moderate',
        'Lower the recorded tensor indexing/slice forms, distinguishing tensor indexing from '
        'tuple projections. Reuse existing slice/gather/broadcast machinery after validating '
        'Python slice bounds, axes, shapes and live-buffer dependencies.'),
    'torch._VariableFunctionsClass.arange': ('Small',
        'Connect the Python factory form to the existing static ATen range factory when its '
        'recorded bounds and shape are static. Preserve dtype/device defaults and integer ranges; '
        'do not infer values from example-dependent runtime inputs.'),
    'torch._VariableFunctionsClass.tensor': ('Small',
        'Lower the observed scalar literal factories with their explicit dtype/device and '
        'fresh-storage/autograd metadata. Reuse scalar word codecs and constant construction.'),
    'torch._VariableFunctionsClass.zeros': ('Small',
        'Normalize the static Python zero factory to existing constant/factory support, retaining '
        'the requested shape, dtype and device.'),
    'new_ones': ('Small',
        'Handle the tensor-method factory using its recorded shape, dtype and device defaults. '
        'Static all-one values can reuse current factories without freezing input values.'),
    'torch._VariableFunctionsClass.diff': ('Moderate',
        'Normalize the recorded integer finite difference to validated slices and signed '
        'subtraction, including dimensions and any boundary operands. Reuse existing integer '
        'arithmetic rather than introducing a floating approximation.'),
    'cumsum': ('Moderate',
        'The current scan rejects the observed dtype conversion. Compose the requested signed '
        'conversion and scan with explicit accumulator/output types and preserve source overflow '
        'and shape behavior. A generic floating cumulative-sum rule is insufficient.'),
    'torchao.choose_qparams_affine.default': ('Moderate',
        'Compose the observed symmetric per-token calibration, float32 epsilon and scale '
        'rounding, and INT8 zero-point result. Preserve the two tensor outputs and their '
        '[batch, token, 1] shapes. Existing absolute/max reductions and casts are ingredients, '
        'not a complete implementation of this custom operator. FloatLib supplies floating '
        'arithmetic; Lean core supplies traversal and signed words.'),
    'torchao.quantize_affine.default': ('Moderate',
        'Implement the observed blockwise float32-to-INT8 conversion using the supplied per-token '
        'scale and zero point. Preserve the actual source division/reciprocal order, ties-to-even '
        'rounding, signed bounds [-128,127] and final cast. Reuse scalar rounding/casts; integer '
        'clamp and the custom-op frontend remain required ingredients.'),
    'export.access_subclass_inner_tensor.default': ('Moderate',
        'Resolve the frozen Int8Tensor payload and float32 scale fields without evaluating '
        'runtime-dependent tensors. The current rules fold the integer payload fields but '
        'refuse the floating scale fields. Add guarded extraction/lifting with actual source '
        'dtype, shape and storage values; this is a representation/payload gap, not a new arithmetic kernel.'),
    'aten.add_.Tensor': ('Moderate',
        'The observed float32 bias additions mutate fresh rescaled projection outputs. Reuse '
        'functional addition after checking freshness, aliases and all later reads. Preserve '
        'source rounding and writes; a blanket in-place-to-functional substitution is insufficient.'),
    'aten.detach_.default': ('Small',
        'Handle the observed fresh frozen float32 scalar following lift_fresh_copy. Reuse '
        'existing detach/value semantics with guarded autograd-metadata and alias checks. '
        'No new numerical kernel is needed for this inference case.'),
    'aten._int_mm.default': ('Hard',
        'Implement INT8 × INT8 → INT32 matrix multiplication, including widening, accumulation and '
        'the source overflow/shape contract. Existing floating GEMM is not an implementation of '
        'this mixed-type operator. Reuse Lean Int/BitVec arithmetic and signed casts; connect the '
        'typed IR, native execution and frontend. Keep integer accumulation ahead of floating rescaling.'),
    'aten.amax.default': ('Moderate',
        'Reuse existing reduction/index traversal, but implement the actual axis/keepdim and dtype '
        'contract. This feeds per-token dynamic scales; preserve edge behavior and intermediate precision.'),
    'aten.abs.default': ('Small',
        'Reuse existing scalar absolute-value semantics and pointwise traversal; add the missing '
        'frontend dispatch only after checking the exact observed dtype and overload.'),
    'aten.clamp.default': ('Small',
        'Preserve signed integer bounds and the source dtype. Reuse signed comparisons and selection '
        'where supported; do not route integer values through a floating-only clamp.'),
    'aten._to_copy.default': ('Moderate',
        'Audit each observed source/destination type and conversion boundary. Reuse supported casts '
        'and FloatLib rounding for floating conversions; check integer bounds and avoid double rounding.'),
    'aten.triu.default': ('Small',
        'Implement static row/column selection at the observed diagonal, or guarded constant folding '
        'when all inputs are frozen. Discarded entries must be positive zero; multiplying by a mask '
        'changes NaN and signed-zero behavior.'),
    'aten.mul_.Tensor': ('Moderate',
        'Reuse multiply after proving the observed mutation rewrite safe using freshness, aliasing '
        'and later-use checks. Do not silently delete writes or change signed-zero multiplication.'),
    'builtins.getattr': ('Hard',
        'Normalize quantized tensor-subclass field access by lifting its INT8 payload and scales '
        'into typed compiler inputs. Preserve the live-parameter contract; do not treat the wrapper '
        'as an ordinary float tensor or freeze runtime-dependent activation values.'),
}


def note(row, category):
    operator = row['operator']
    if category == 'export' and operator == '_operator.getitem':
        return ('Small',
            'Connect projections of the two-output TorchAO calibration operator to its actual '
            'typed output ports. Tuple metadata alone is not a recognized lowering: the current '
            'runtime refuses these getitems. Preserve each output shape and dtype, and only '
            'route projections once the producer is supported. No new arithmetic kernel is needed.')
    if operator in NOTES:
        return NOTES[operator]
    if category == 'compile':
        return ('Moderate',
            'Audit this Dynamo/Python operator form and normalize it to the corresponding supported '
            'ATen contract where valid. Compare its arguments, metadata and type promotion with the '
            'export path. A missing frontend spelling does not establish a missing numeric kernel.')
    return ('Moderate',
        'Implement or connect the observed contract shown in the node evidence. Reuse existing '
        'typed operations and codecs where their semantics match. Check emitter/payload requirements '
        'and run native execution before treating local rule recognition as completed support.')


def rank_gaps(features):
    rows = []
    for category, inventory in features.items():
        sequence = []
        for row in inventory:
            if not row['unsupported']:
                continue
            difficulty, guidance = note(row, category)
            sequence.append({**row, 'category': category, 'difficulty': difficulty,
                             'guidance': guidance})
        sequence.sort(key=lambda r: ({'Hard': 0, 'Moderate': 1, 'Small': 2}[r['difficulty']], r['operator']))
        for i, row in enumerate(sequence, 1):
            row['id'] = ('A' if category == 'export' else 'F') + str(i).zfill(2)
            rows.append(row)
    return rows


def esc(value):
    return html.escape(str(value), quote=True)


def details(label, value):
    return '<details><summary>' + esc(label) + '</summary><pre>' + esc(json.dumps(value, indent=2)) + '</pre></details>'


def render_html(report):
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>' + esc(report['title']) + '</title><style>',
        'body{margin:0;background:#f3f6fa;color:#192c40;font:16px/1.55 system-ui,sans-serif}',
        'main{max-width:1420px;margin:auto;padding:38px}h1{font-size:36px;line-height:1.2}',
        'h2{margin-top:36px}a{color:#125d81}code,pre{font-family:ui-monospace,monospace}',
        'code{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}',
        '.notice{background:#fff1cf;border-left:4px solid #b57c00;padding:16px}',
        '.stats{display:flex;gap:12px;flex-wrap:wrap}.stats p{background:white;padding:14px;border:1px solid #dce4ee;border-radius:8px}',
        '.stats b{font-size:24px}.table{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}',
        'th,td{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #dce4ee}th{background:#e4ecf5}',
        'details{margin:10px 0}summary{cursor:pointer}tr:target{background:#fff2bd}',
        '@media print{main{padding:0}body{background:white}table{font-size:10px}th,td{padding:6px}}',
        '</style></head><body><main><p>TorchLean · TorchAO W8A8 · measured diagnostic</p>',
        '<h1>' + esc(report['title']) + '</h1>',
        '<p>Runtime baseline: <code>' + esc(report['runtime_revision']) + '</code></p>',
        '<p class="notice"><b>Full-model bitwise equivalence: not established.</b> '
        'This report separates the ATen/TorchAO export graph, the public compiler frontend, small native '
        'fixtures and whole-model conversion. Local rule recognition is not full conversion, '
        'execution, FPGA acceptance or a proof. Source graphs are inventoried without replacing '
        'unsupported computations by stubs.</p>',
        '<p><a href="#ranked">Export gaps</a> · <a href="#frontend">Compiler frontend gaps</a> · '
        '<a href="#fixtures">Native fixtures</a> · <a href="#coverage">Complete operator inventory</a> · '
        '<a href="#provenance">Provenance</a></p>']
    parts.append('<div class="stats">')
    for category, inventory in report.get('inventories', {}).items():
        features = report.get('features', {}).get(category, [])
        parts.append('<p><b>' + str(inventory.get('node_count', 0)) + '</b> ' + esc(category) + ' source nodes</p>')
        parts.append('<p><b>' + str(sum(bool(r['unsupported']) for r in features)) + '</b> ' + esc(category) + ' failing operator forms</p>')
        parts.append('<p><b>' + str(sum(r['probe_errors'] for r in features)) + '</b> ' + esc(category) + ' diagnostic errors</p>')
    parts.append('</div><h2>Target computation</h2>')
    parts.append(details('Pinned model, quantization and input/output contract', report.get('model', {})))
    parts.append('<p>Primary target: symmetric INT8 weights per output channel, symmetric dynamic '
        'INT8 activations per token, INT32 dot products followed by floating rescaling. Embeddings, '
        'attention and normalization retain floating work. This is a new numerical target relative '
        'to the old PT2E per-tensor affine archive. A newer quantizer does not establish language '
        'quality or FPGA performance; both require separate measurements.</p>')
    parts.append('<h2>Whole-model entry points</h2><div class="table"><table><tr><th>Path</th><th>Observed outcome</th><th>Evidence</th></tr>')
    for category, result in report['whole_model'].items():
        parts.append('<tr><td><code>' + esc(report.get('paths', {}).get(category, category)) + '</code></td><td><b>' +
            esc(result['status']) + '</b><br>' + esc(result.get('message', '')) + '</td><td>' + details('Actual result / traceback', result) + '</td></tr>')
    parts.append('</table></div><p>The frontend inventory is collected at the ordinary compiler '
        'lowering boundary and deliberately stopped after inspection. That diagnostic run is '
        'not a successful compiled execution.</p>')
    semantics = report.get('whole_model', {}).get('compile', {}).get('quantized_representation')
    if semantics:
        parts.append('<p class="notice"><b>F00: quantized semantics are not represented by a floating '
            'linear rule.</b> ' + esc(semantics['message']) + '</p>')
    for category, anchor, title in [('export', 'ranked', 'Remaining exported-graph gaps, hardest first'),
                                     ('compile', 'frontend', 'Additional public compiler frontend gaps')]:
        parts.append('<h2 id="' + anchor + '">' + title + '</h2>')
        parts.append('<p>Difficulty is an engineering estimate for the recorded signatures. '
            'A-IDs refer to exported operator contracts; F-IDs refer to compiler forms. They are '
            'separate workloads and must not be added into a universal gap count. Repairing one '
            'may expose further conversion, execution or numerical failures.</p>')
        parts.append('<div class="table"><table><tr><th>ID</th><th>Operator and calls</th><th>Difficulty</th><th>Implementation work / reuse</th><th>Measured contracts</th></tr>')
        ordered = sorted(report['gaps'], key=lambda row: (row.get('status') != 'representation_gap',
                         {'Hard': 0, 'Moderate': 1, 'Small': 2}.get(row['difficulty'], 3), row['id']))
        for row in ordered:
            if row['category'] != category:
                continue
            parts.append('<tr id="gap-' + esc(row['id'].lower()) + '"><td>' + esc(row['id']) + '</td><td><code>' +
                esc(row['operator']) + '</code><br>' +
                (str(row['calls']) + ' calls with a semantic representation gap' if row.get('status') == 'representation_gap'
                 else str(row.get('unsupported', row['calls'])) + ' refused / ' + str(row['calls']) + ' calls') +
                '</td><td>' + esc(row['difficulty']) + '</td><td>' + esc(row['guidance']) + '</td><td>' +
                details('Representative source nodes; all nodes are in the evidence JSON', row.get('examples', [])) + '</td></tr>')
        parts.append('</table></div>')
    parts.append('<h2>Library reuse and FPGA consequences</h2><p>Lean core Int/BitVec and tensor '
        'indexing supply the signed arithmetic and traversal ingredients. FloatLib supplies '
        'existing floating arithmetic, rounding and storage words. Mathlib or CSLib may help '
        'later specifications and proofs; they are not substitutes for missing executable '
        'PyTorch operators. Implementing integer dot products and preserving quantization '
        'boundaries are necessary for the intended W8A8 FPGA datapath. Floating attention, '
        'normalization and rescaling remain part of this target. Operator counts alone do not '
        'establish speed, area, timing closure or language quality.</p>')
    parts.append('<h2 id="fixtures">Native fixture results</h2><p>Small real examples use the native '
        'checker and eager PyTorch reference. Passing one fixture establishes that example only; '
        'it does not resolve every source signature in the model.</p><div class="table"><table><tr><th>Fixture</th><th>Outcome</th><th>Native report / failure</th></tr>')
    for fixture in report['fixtures']:
        parts.append('<tr><td>' + esc(fixture['name']) + '</td><td>' + esc(fixture['status']) + '</td><td>' + details('Evidence', fixture) + '</td></tr>')
    parts.append('</table></div><h2 id="coverage">Complete operator inventory</h2>')
    for category, features in report.get('features', {}).items():
        source = report['inventories'][category]
        parts.append('<h3>' + esc(category) + '</h3><p>' + str(source['node_count']) +
            ' nodes including declarations and output; ' + str(sum(r['calls'] for r in features)) +
            ' operator nodes. Every source node, argument, dependency, result metadata and probe '
            'outcome is retained in the JSON evidence. Frozen/runtime dependency is recorded '
            'for the exported graph; compiler placeholders remain live inputs.</p>')
        parts.append(details('Graph-wide preprocessing diagnostics', source.get('preprocessing_probes', [])))
        parts.append('<div class="table"><table><tr><th>Operator</th><th>Calls</th><th>Recognized locally</th><th>Refused</th><th>Diagnostic errors</th><th>Unchecked</th></tr>')
        for row in features:
            parts.append('<tr><td><code>' + esc(row['operator']) + '</code></td>' +
                         ''.join('<td>' + str(row[k]) + '</td>' for k in ('calls', 'recognized', 'unsupported', 'probe_errors', 'unchecked')) + '</tr>')
        parts.append('</table></div>')
    parts.append('<h2>Warnings and limitations</h2><p>Bulk tensor payloads are omitted from local '
        'rule probes; ordinary conversion uses the unmodified importer and real parameters. '
        'Native fixture expectations are eager outputs, not universal proofs. Full-model native '
        'execution, full-model bitwise agreement, hardware synthesis, latency, resource use and '
        'held-out language quality are not established by this survey.</p>')
    parts.append(details('Recorded upstream warnings', report.get('warnings', [])))
    parts.append('<h2>Reproduce</h2><pre>' + esc(report['reproduction']) + '</pre>')
    parts.append('<h2 id="provenance">Provenance</h2>' + details('Package versions, runtime/diagnostic/checker hashes', report['provenance']))
    parts.append('<p>Machine-readable evidence: <a href="TINYSTORIES_TORCHAO_SURVEY_EVIDENCE.json">'
                 'TINYSTORIES_TORCHAO_SURVEY_EVIDENCE.json</a>. Reproduction: '
                 '<a href="TINYSTORIES_TORCHAO_SURVEY.md">TINYSTORIES_TORCHAO_SURVEY.md</a>.</p></main></body></html>')
    return '\n'.join(parts) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render_html(json.loads(args.report.read_text())))
