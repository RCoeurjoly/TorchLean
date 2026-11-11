"""Read-only inventory using TorchLean rules, adapted from the prior pinned survey.

A recognized local rule is not a successful full conversion or an execution claim.
Bulk payload serialization is omitted only in this diagnostic; ordinary conversion
runs separately with real weights and the unmodified runtime.
"""

import contextlib
import copy

class CurrentAdapter:
    """Read current capture rules; preserve their shared dictionary identities."""

    def __init__(self):
        from torchlean import _fx, _graph, _emit, _pointwise, _structure
        self.fx, self.graph, self.emit = _fx, _graph, _emit
        self.pointwise, self.structure = _pointwise, _structure
        self._ARITHMETIC_KINDS = _pointwise._ARITHMETIC_KINDS
        self._tensor_values = lambda _: []  # Legacy inspector hook; current capture uses words.

    @property
    def _TENSOR_VALUES(self):
        return dict(self.fx._TENSOR_VALUES)

    @_TENSOR_VALUES.setter
    def _TENSOR_VALUES(self, values):
        self.fx._TENSOR_VALUES.clear()
        self.fx._TENSOR_VALUES.update(values)

    @property
    def _tensor_bits(self):
        return self.fx._tensor_bits

    @_tensor_bits.setter
    def _tensor_bits(self, function):
        self.fx._tensor_bits = function

    def __getattr__(self, name):
        if name in ("_dtype_of", "_tuple_shapes_of_node", "_zeros_like"):
            return getattr(self.fx, name)
        if name in ("_routing_helpers", "_metadata_constraint"):
            return getattr(self.graph, name)
        raise AttributeError(name)

    def _tuple_kind(self, node, model):
        if self.structure._component_producer(node):
            return "component_producer"
        raise NotImplementedError(f"unsupported tuple-valued PyTorch op: {node.target}")

    def _lower_kind(self, node, model):
        result = self.emit._lower_kind(node, model)
        if "_alias" in result:
            return {**result, "kind": "alias"}
        if "_constant" in result:
            return {**result, "kind": "constant"}
        return result

    def _convert(self, nodes, parent, dtype):
        return self.pointwise._cast_to(nodes, parent, dtype)

    def _emit_pointwise(self, plan, nodes, bindings, shape, dtype):
        emitter = (self.emit._emit_structure if plan["op"] in self.structure._STRUCTURE_PLANS
                   else self.pointwise._emit_pointwise)
        return emitter(plan, nodes, bindings, shape, dtype)


@contextlib.contextmanager
def capture_state(program, checker):
    adapter = CurrentAdapter()
    dictionaries = [(adapter.fx, "_TENSOR_VALUES")]
    dictionaries += [(adapter.structure, name) for name in
                     ("_FOLDED", "_BUFFERS", "_OUTPUT_TARGETS", "_BUFFER_UPDATES", "_DROPOUT_MASKS")]
    with adapter.graph._LOCK:
        saved = [(module, name, copy.copy(getattr(module, name))) for module, name in dictionaries]
        saved_checker = adapter.fx._CHECKER[:]
        saved_bits = adapter.fx._tensor_bits
        try:
            adapter.graph._reset(program.graph_signature if program is not None else None, str(checker))
            if program is not None:
                state, constants = dict(program.state_dict), dict(program.constants)
                import torch
                for spec in program.graph_signature.input_specs:
                    if spec.kind.name != "USER_INPUT":
                        value = state.get(spec.target, constants.get(spec.target))
                        if isinstance(value, torch.Tensor):
                            adapter.fx._TENSOR_VALUES[spec.arg.name] = value
            yield adapter
        finally:
            adapter.fx._tensor_bits = saved_bits
            adapter.fx._CHECKER[:] = saved_checker
            for module, name, contents in saved:
                value = getattr(module, name)
                value.clear()
                value.update(contents) if isinstance(value, dict) else value.extend(contents)


def inspect_export(program, adapter):
    """Inspect the original source graph without serializing bulk payloads or normalizing effects."""
    import operator
    import torch

    before_count = len(list(program.graph.nodes))
    by_name = {node.name: node for node in program.graph.nodes}
    specs = {spec.arg.name: spec for spec in program.graph_signature.input_specs}
    state, constants = dict(program.state_dict), dict(program.constants)
    saved_tensor_values = adapter._TENSOR_VALUES
    tensor_values = {}
    for name, spec in specs.items():
        if spec.kind.name != "USER_INPUT":
            value = state.get(spec.target, constants.get(spec.target))
            if isinstance(value, torch.Tensor):
                tensor_values[name] = value

    def descriptor(value):
        if isinstance(value, torch.Tensor) or hasattr(value, "shape") and hasattr(value, "dtype"):
            if any(type(dim) is not int for dim in value.shape):
                raise ValueError("symbolic dimensions need a symbolic interface; this survey requires static shapes")
            return {"kind": "tensor", "shape": list(value.shape),
                    "dtype": adapter._dtype_of(value), "device": str(getattr(value, "device", "cpu"))}
        if isinstance(value, (tuple, list)):
            return {"kind": "tuple" if isinstance(value, tuple) else "list",
                    "items": [descriptor(child) for child in value]}
        if value is None:
            return {"kind": "none"}
        return {"kind": "unknown", "type": type(value).__name__}

    def result(node):
        if "val" in node.meta:
            return descriptor(node.meta["val"])
        if "tensor_meta" in node.meta:
            return descriptor(node.meta["tensor_meta"])
        schema = getattr(node.target, "_schema", None)
        if schema is not None and not schema.returns:
            return {"kind": "none"}
        return {"kind": "unknown", "reason": "source has no result metadata"}

    def encode(value):
        if isinstance(value, torch.fx.Node):
            return {"node": value.name}
        if isinstance(value, torch.Tensor):
            return {"tensor_literal_metadata": descriptor(value)}
        if isinstance(value, (list, tuple)):
            return [encode(child) for child in value]
        if isinstance(value, dict):
            return {str(key): encode(child) for key, child in value.items()}
        if isinstance(value, float):
            import math
            return value if math.isfinite(value) else {"float": repr(value)}
        if value is None or isinstance(value, (str, int, bool)):
            return value
        return str(value)

    def references(value):
        if isinstance(value, torch.fx.Node):
            return [value.name]
        if isinstance(value, (list, tuple)):
            return [name for child in value for name in references(child)]
        if isinstance(value, dict):
            return [name for child in value.values() for name in references(child)]
        if value is None:
            return []
        raise ValueError("non-tensor source outputs need a separate model interface")

    saved_bits, saved_values = adapter._tensor_bits, adapter._tensor_values
    nodes, outputs, runtime, preprocessing = [], [], {}, []
    try:
        adapter._TENSOR_VALUES = tensor_values
        # Some scalar words are configuration (notably calibration epsilon), not bulk payload.
        adapter._tensor_bits = lambda value: saved_bits(value) if value.numel() <= 1 else []
        adapter._tensor_values = lambda value: []
        try:
            helpers = adapter._routing_helpers(program.graph)
        except Exception as error:
            helpers = set()
            preprocessing.append({"stage": "routing_helpers",
                "status": "unsupported" if isinstance(error, NotImplementedError) else "probe_error",
                "error_type": type(error).__name__, "message": str(error),
                "scope": "graph-wide recognizer failed; local node probes do not establish this pattern's support"})
        for node in program.graph.nodes:
            dependencies = [parent.name for parent in node.all_input_nodes]
            spec = specs.get(node.name)
            role = spec.kind.name.lower() if spec is not None else None
            record = {"id": node.name, "op": node.op, "target": str(node.target),
                      "dependencies": dependencies, "arguments": encode(node.args),
                      "keywords": encode(node.kwargs), "role": role,
                      "mutable": bool(getattr(getattr(node.target, "_schema", None), "is_mutable", False)),
                      "effect_semantics": "recorded_not_proved"}
            runtime[node.name] = role == "user_input" or any(runtime[name] for name in dependencies)
            record["depends_on_runtime_input"] = runtime[node.name]
            stack = node.meta.get("nn_module_stack", {})
            record["modules"] = [str(value[0]) for value in stack.values() if isinstance(value, tuple)]
            if node.op == "output":
                outputs = references(node.args[0])
                nodes.append(record)
                continue
            record["result"] = result(node)
            if node.target is operator.getitem and len(node.args) == 2 and isinstance(node.args[0], torch.fx.Node):
                origin = result(node.args[0])
                if origin["kind"] in ("tuple", "list") and type(node.args[1]) is int:
                    record["projection"] = {"source": node.args[0].name, "index": node.args[1]}
            if node.op in ("placeholder", "get_attr"):
                record["probe"] = {"status": "declaration", "payload": "omitted"}
            else:
                try:
                    if "projection" in record:
                        # Tuple metadata records an interface, not implementation support.
                        # Only the current producer/projection rule can establish recognition.
                        lowered = adapter._lower_kind(node, None)
                        detail = {"kind": lowered.get("kind"), "plan": lowered.get("_plan", {}).get("op")}
                    elif node in helpers or adapter._zeros_like(node):
                        detail = {"kind": "routing_helper"}
                    elif getattr(adapter, "_qparams_plan", lambda _: None)(node) is not None:
                        detail = {"kind": "choose_qparams_int8"}
                    elif adapter._metadata_constraint(node) is not None:
                        detail = {"kind": "metadata_constraint", "constraint_status": "not_checked"}
                    elif adapter._tuple_shapes_of_node(node):
                        detail = {"kind": adapter._tuple_kind(node, None)}
                    else:
                        lowered = adapter._lower_kind(node, None)
                        detail = {"kind": lowered.get("kind"), "plan": lowered.get("_plan", {}).get("op")}
                        plan = lowered.get("_plan")
                        ordinary_arithmetic = lowered.get("kind") in (
                            *adapter._ARITHMETIC_KINDS.values(), "div", "max_elem", "min_elem")
                        if (plan and plan["op"] not in ("triu", "greater", "where")) or (not plan and ordinary_arithmetic):
                            tensor_parents = [parent for parent in node.all_input_nodes
                                              if result(parent)["kind"] == "tensor"]
                            temporary = [{"id": index, "kind": "input", "parents": [],
                                          "shape": result(parent)["shape"], "dtype": result(parent)["dtype"]}
                                         for index, parent in enumerate(tensor_parents)]
                            if plan:
                                adapter._emit_pointwise(plan, temporary,
                                    dict(zip(tensor_parents, range(len(tensor_parents)))),
                                    record["result"]["shape"], record["result"]["dtype"])
                            else:
                                for index in range(len(tensor_parents)):
                                    adapter._convert(temporary, index, record["result"]["dtype"])
                            detail["implicit_casts"] = [
                                {"source_dtype": temporary[item["parents"][0]]["dtype"], "dtype": item["dtype"]}
                                for item in temporary if item.get("primitive") == "cast"]
                    record["probe"] = {"status": "recognized", **detail,
                                       "scope": "rule recognition; payload words and execution not checked"}
                except Exception as error:
                    record["probe"] = {"status": "unsupported" if isinstance(error, NotImplementedError) else "probe_error",
                                       "error_type": type(error).__name__,
                                       "message": str(error)}
            nodes.append(record)
    finally:
        adapter._tensor_bits, adapter._tensor_values = saved_bits, saved_values
        adapter._TENSOR_VALUES = saved_tensor_values
    from torch.utils._pytree import treespec_dumps

    def signature(spec):
        return {"name": getattr(spec.arg, "name", None), "argument": str(spec.arg),
                "kind": spec.kind.name, "target": spec.target,
                "persistent": getattr(spec, "persistent", None)}

    call_spec = getattr(program, 'call_spec', None)
    return {"inputs": [spec.arg.name for spec in program.graph_signature.input_specs
                        if spec.kind.name == "USER_INPUT"], "outputs": outputs, "nodes": nodes,
            "call_spec": ({"format": "torch.utils._pytree.treespec",
                          "in_spec": treespec_dumps(program.call_spec.in_spec),
                          "out_spec": treespec_dumps(program.call_spec.out_spec)}
                          if call_spec is not None else {"format": "flat compiler backend graph"}),
            "range_constraints": {str(k): str(v) for k, v in getattr(program, 'range_constraints', {}).items()},
            "graph_signature": {"inputs": [signature(s) for s in program.graph_signature.input_specs],
                                "outputs": [signature(s) for s in program.graph_signature.output_specs]},
            "preprocessing_probes": preprocessing,
            "canonicalization": {"before_nodes": before_count, "after_nodes": len(nodes),
                                 "method": "none; original exported graph"},
            "payload_words": "omitted", "semantics": "not_executed"}
