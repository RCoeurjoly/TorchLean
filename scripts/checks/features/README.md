# PyTorch feature regressions

Each `test_*.py` file covers one feature and runs independently. A feature PR can
include that single test file and its implementation changes. The tests use the
existing TorchLean Python package and checker; they do not import another new test
file, a shared backlog helper, or a model-specific survey.

These files exercise different capture paths. For the exported and decomposed
TinyStories inference graph, see the [Core ATen gap report](../../../docs/TINYSTORIES_CORE_ATEN_GAPS.html).
Its current gaps are Boolean-data advanced indexing in the frontend and INT32
matrix multiplication in native IR import. The export normalizes `_int_mm` to
Core `mm` with explicit INT32 casts; the report also records native numerical differences. The remaining files
are API coverage tests and are not all requirements of that export pipeline.

These are pending regressions: they assert successful support and currently fail
where support is missing. They are ordinary `unittest` tests with no expected-failure
annotations or special runner. Once a feature works, its file becomes a passing
regression test without changing the assertions. The directory is opt-in and is not
automatically run by `scripts/lake.sh test`.

Run from the repository root using a Python environment with PyTorch and the editable
TorchLean package installed. Build the checker with `scripts/lake.sh build verify`,
or set `TORCHLEAN_VERIFY` to an existing checker executable.

```bash
source .venv/bin/activate
python scripts/checks/features/test_pytorch_embedding.py -v
# Select a case within that feature, if needed:
python scripts/checks/features/test_pytorch_embedding.py \
  EmbeddingTests.test_compiled_embedding_keeps_weights_live -v
```

The embedding file also checks the already supported export path. Compiler cases
compare the captured IR's native execution with eager PyTorch, as well as checking
framework execution. Embedding and layer-normalization cases change parameters
between calls so accidentally frozen weights cannot pass.

| Test file | Feature | Dependencies beyond PyTorch and TorchLean |
| --- | --- | --- |
| [test_aten_int_mm.py](test_aten_int_mm.py) | INT8 matrix multiplication with INT32 results | None |
| [test_aten_add_inplace.py](test_aten_add_inplace.py) | In-place bias addition on a fresh tensor | None |
| [test_aten_detach_inplace.py](test_aten_detach_inplace.py) | In-place detach of a fresh frozen scalar | None |
| [test_torchao_tensor_scale.py](test_torchao_tensor_scale.py) | Frozen tensor-subclass scale fields | TorchAO |
| [test_torchao_choose_qparams_affine.py](test_torchao_choose_qparams_affine.py) | Per-token symmetric calibration | TorchAO |
| [test_torchao_quantize_affine.py](test_torchao_quantize_affine.py) | Affine quantization, rounding and saturation | TorchAO |
| [test_pytorch_tuple_projection.py](test_pytorch_tuple_projection.py) | Scale and zero-point tuple projections | TorchAO; also needs `choose_qparams_affine` support |
| [test_torchao_compiled_linear.py](test_torchao_compiled_linear.py) | Compiled W8A8 linear integer semantics | TorchAO |
| [test_pytorch_layer_norm.py](test_pytorch_layer_norm.py) | Layer normalization with live affine parameters | None |
| [test_pytorch_embedding.py](test_pytorch_embedding.py) | Embedding lookup with live weights | None |
| [test_pytorch_tensor_slicing.py](test_pytorch_tensor_slicing.py) | Tensor slicing | None |
| [test_pytorch_bool_advanced_index.py](test_pytorch_bool_advanced_index.py) | Core ATen advanced indexing of Boolean data by INT64 tensors | None |
| [test_pytorch_bool_cumsum.py](test_pytorch_bool_cumsum.py) | Boolean cumulative sum promoted to INT64 | None |
| [test_pytorch_diff.py](test_pytorch_diff.py) | Integer differences with prepend | None |
| [test_pytorch_new_ones.py](test_pytorch_new_ones.py) | Scalar boolean `new_ones` | None |
| [test_pytorch_arange.py](test_pytorch_arange.py) | Static integer `arange` | None |
| [test_pytorch_scalar_tensor.py](test_pytorch_scalar_tensor.py) | Float32 scalar tensor construction | None |
| [test_pytorch_scalar_zeros.py](test_pytorch_scalar_zeros.py) | Float32 scalar zeros | None |

The TorchAO cases were exercised with CPU PyTorch 2.15 and TorchAO 0.18 using
`Int8DynamicActivationInt8WeightConfig(version=2)` where a quantized layer is needed.
They download no model. The W8A8 linear test checks numerical agreement and the
presence of INT8 payloads and INT32 computation in the checked IR; structure and
type checks alone cannot establish quantized support. Its float-linear control
checks the same native replay path without quantization.
