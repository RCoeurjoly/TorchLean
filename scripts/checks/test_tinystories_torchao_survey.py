"""Evidence-integrity checks for the TorchAO TinyStories diagnostic."""

import copy
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import torch

from tinystories_torchao_survey import inspect_program, summarize, render_html, compile_path, validate_checkpoint


class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lookup = torch.nn.Embedding(8, 4)

    def forward(self, tokens):
        return self.lookup(tokens)


class IntegerMM(torch.nn.Module):
    def forward(self, a, b):
        return torch._int_mm(a, b)


class SurveyTests(unittest.TestCase):
    checker = Path(__file__).resolve().parents[2] / '.lake/build/bin/verify'

    def embedding_program(self):
        return torch.export.export(Embedding().eval(), (torch.tensor([[0, 3, 7]]),))

    def test_inventory_accounts_for_every_source_node_and_dependency(self):
        program = self.embedding_program()
        before = str(program.graph)
        source = inspect_program(program, self.checker)
        self.assertEqual(len(source['nodes']), len(list(program.graph.nodes)))
        self.assertEqual(str(program.graph), before)
        seen = set()
        for node in source['nodes']:
            self.assertTrue(set(node['dependencies']) <= seen)
            self.assertNotIn(node['id'], seen)
            seen.add(node['id'])
        features = summarize(source)
        embedding = next(row for row in features if row['operator'] == 'aten.embedding.default')
        self.assertEqual(embedding['unsupported'], 0)
        self.assertEqual(embedding['recognized'], 1)
        self.assertEqual(sum(row['calls'] for row in features),
                         sum(n['op'] not in ('placeholder', 'get_attr', 'output') for n in source['nodes']))

    def test_integer_matmul_refusal_is_observed(self):
        a = torch.ones(8, 8, dtype=torch.int8)
        program = torch.export.export(IntegerMM(), (a, a))
        rows = summarize(inspect_program(program, self.checker))
        row = next(r for r in rows if r['operator'] == 'aten._int_mm.default')
        self.assertEqual(row['unsupported'], 1)
        self.assertIn('unsupported PyTorch op', row['examples'][0]['probe']['message'])

    def test_compile_frontend_inventory_survives_the_real_entry_failure(self):
        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, tokens):
                return self.inner(tokens)

        result, source = compile_path(Wrapper(Embedding().eval()), torch.tensor([[0, 3, 7]]), self.checker)
        self.assertEqual(result['status'], 'refused')
        self.assertIn('embedding', result['message'])
        self.assertTrue(source, result.get('inventory_error'))
        self.assertTrue(any('embedding' in row['operator'] for row in summarize(source)))

    def test_unexpected_compiler_failure_is_not_mislabeled_as_missing_support(self):
        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

        def broken(*args, **kwargs):
            raise ValueError('unexpected compiler failure')

        with patch('torchlean.compile', return_value=broken):
            result, source = compile_path(Wrapper(Embedding()), torch.tensor([[0]]), self.checker)
        self.assertEqual(result['status'], 'error')
        self.assertFalse(source)

    def test_custom_tuple_metadata_does_not_imply_projection_support(self):
        import torchao

        class Choose(torch.nn.Module):
            def forward(self, x):
                return torch.ops.torchao.choose_qparams_affine.default(
                    x, 'SYMMETRIC', [1, 1, 4], torch.int8, -128, 127,
                    torch.finfo(torch.float32).eps, torch.float32, torch.int8, True)

        program = torch.export.export(Choose(), (torch.ones(1, 2, 4),))
        rows = summarize(inspect_program(program, self.checker))
        projections = next(row for row in rows if row['operator'] == '_operator.getitem')
        self.assertEqual(projections['calls'], 2)
        self.assertEqual(projections['unsupported'], 2)
        self.assertEqual(projections['recognized'], 0)

    def test_internal_diagnostic_errors_are_not_counted_as_missing_support(self):
        program = self.embedding_program()
        with patch('torchlean._emit._lower_kind', side_effect=ValueError('diagnostic broke')):
            rows = summarize(inspect_program(program, self.checker))
        row = next(r for r in rows if r['operator'] == 'aten.embedding.default')
        self.assertEqual(row['probe_errors'], 1)
        self.assertEqual(row['unsupported'], 0)

    def test_html_escapes_evidence_and_states_full_model_failure(self):
        report = {
            'title': 'TinyStories <test>', 'runtime_revision': 'abc',
            'provenance': {}, 'model': {}, 'paths': {}, 'fixtures': [],
            'gaps': [{'id': 'E01', 'operator': '<script>bad()</script>', 'difficulty': 'Hard',
                      'calls': 1, 'category': 'export', 'guidance': 'missing <kernel>',
                      'examples': [], 'probe_errors': 0}],
            'whole_model': {'export': {'status': 'refused', 'message': '<bad>'},
                            'compile': {'status': 'refused', 'message': 'embedding'}},
            'inventories': {}, 'reproduction': 'python survey.py', 'warnings': [],
        }
        document = render_html(report)
        self.assertNotIn('<script>bad()</script>', document)
        self.assertIn('&lt;script&gt;bad()', document)
        self.assertIn('refused', document)
        self.assertIn('not established', document)

    def test_different_checkpoint_cannot_be_mislabeled_as_pinned_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config.json').write_text('{}')
            (root/'pytorch_model.bin').write_bytes(b'different weights')
            with self.assertRaisesRegex(ValueError, 'differs from the pinned'):
                validate_checkpoint(root)


if __name__ == '__main__':
    unittest.main()
