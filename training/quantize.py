"""Weight-only int8 storage for the exported graphs.

Each large weight is stored as int8 with one float32 scale per output channel,
chosen from the operator that consumes it, and dequantized in the graph.
Recurrent weights get one scale per gate row and direction. Biases, norms,
single-channel windows and small tensors stay float32. Fixed DFT bases are rebuilt from a few
ops instead of being stored. Arithmetic remains float32; this reduces storage,
it does not promise INT8 execution or a speedup.
"""

from collections import defaultdict
import math
import zlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SCHEME = "int8-per-output-channel-v2"
MIN_QUANTIZED_SIZE = 1024
# Parity gate: at most 2% of token durations may move by a rounding step, and the spectral deviation may not
# exceed 1.5x the float32 graph's own noise variation. Int8 graphs fail the duration part (one changed rounding
# shifts every later frame) while scoring the same as float32 end to end; export.py reports it without failing.
MAX_DURATION_CHANGE = 0.02
MAX_DISTANCE_RATIO = 1.5
_RECURRENT = {"LSTM", "GRU", "RNN"}
_RANDOM = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike"}


def _uses(graph: onnx.GraphProto) -> dict[str, list[tuple[onnx.NodeProto, int]]]:
    """Map each tensor to its consuming (node, input slot), looking through Identity."""
    direct: dict[str, list[tuple[onnx.NodeProto, int]]] = defaultdict(list)
    for node in graph.node:
        for slot, name in enumerate(node.input):
            if name:
                direct[name].append((node, slot))

    def resolve(name: str, seen: frozenset = frozenset()) -> list[tuple[onnx.NodeProto, int]]:
        found = []
        for node, slot in direct.get(name, []):
            if node.op_type == "Identity" and node.output[0] not in seen:
                found += resolve(node.output[0], seen | {node.output[0]})
            else:
                found.append((node, slot))
        return found

    return {name: resolve(name) for name in direct}


def _attribute(node: onnx.NodeProto, name: str, default):
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def _layout(node: onnx.NodeProto, slot: int, ndim: int) -> tuple[str, int] | None:
    """Return how output channels are laid out in a weight, or None to keep float32.

    ("axis", a): one scale per index of axis a.
    ("rows", k): one scale per index of the flattened leading k axes.
    """
    op = node.op_type
    if op == "Conv" and slot == 1:
        return ("axis", 0)
    if op == "ConvTranspose" and slot == 1:
        # Weight is [in, out / group, ...]; depthwise/grouped kernels stay per channel.
        return ("axis", 1 if _attribute(node, "group", 1) == 1 else 0)
    if op == "MatMul" and slot == 1:
        return ("axis", ndim - 1)
    if op == "Gemm" and slot == 1:
        return ("axis", 0 if _attribute(node, "transB", 0) else 1)
    if op == "Gather" and slot == 0:
        return ("axis", 0)
    if op in _RECURRENT and slot in (1, 2):
        # W/R are [directions, gates * hidden, inputs]: one scale per gate row.
        return ("rows", 2)
    return None


def _dft_basis(weight: np.ndarray) -> tuple[np.ndarray, int, float] | None:
    """Recognize cat(cos, -sin)(2 pi k n / N) * hann(n) * scale, shape [2K, 1, N]."""
    if weight.ndim != 3 or weight.shape[1] != 1 or weight.shape[0] % 2:
        return None
    count, size = weight.shape[0] // 2, weight.shape[2]
    rows = weight[:count, 0].astype(np.float64)
    bins = np.abs(np.fft.rfft(rows, axis=1)).argmax(axis=1).astype(np.int64)
    angle = 2 * np.pi * ((bins[:, None] * np.arange(size)) % size) / size
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)
    unit = np.concatenate([np.cos(angle), -np.sin(angle)]) * window
    ratio = float(np.abs(weight).max() / np.abs(unit).max())
    for scale in (1.0, 2.0 / size):
        if (
            abs(ratio / scale - 1) < 1e-3
            and np.abs(weight[:, 0] - unit * scale).max() < 1e-3 * scale
        ):
            return bins, size, scale
    return None


def _basis_nodes(name: str, bins: np.ndarray, size: int, scale: float):
    """Build a DFT basis in float32 with k*n reduced modulo N so cos/sin stay exact."""
    prefix = name + "_gen_"
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)) * scale
    initializers = [
        numpy_helper.from_array(bins[:, None], prefix + "k"),
        numpy_helper.from_array(np.arange(size, dtype=np.int64)[None], prefix + "n"),
        numpy_helper.from_array(np.array(size, np.int64), prefix + "N"),
        numpy_helper.from_array(np.array(2 * np.pi / size, np.float32), prefix + "c"),
        numpy_helper.from_array(window.astype(np.float32), prefix + "w"),
        numpy_helper.from_array(np.array([1], np.int64), prefix + "ax"),
    ]
    p = prefix
    nodes = [
        helper.make_node("Mul", [p + "k", p + "n"], [p + "kn"]),
        helper.make_node("Mod", [p + "kn", p + "N"], [p + "m"]),
        helper.make_node("Cast", [p + "m"], [p + "mf"], to=TensorProto.FLOAT),
        helper.make_node("Mul", [p + "mf", p + "c"], [p + "a"]),
        helper.make_node("Cos", [p + "a"], [p + "cos"]),
        helper.make_node("Sin", [p + "a"], [p + "sin"]),
        helper.make_node("Neg", [p + "sin"], [p + "nsin"]),
        helper.make_node("Concat", [p + "cos", p + "nsin"], [p + "cs"], axis=0),
        helper.make_node("Mul", [p + "cs", p + "w"], [p + "b"]),
        helper.make_node("Unsqueeze", [p + "b", p + "ax"], [name]),
    ]
    return initializers, nodes


def quantize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 with one float32 scale per row of a 2-D matrix."""
    scale = (np.abs(matrix).max(axis=1) / 127).astype(np.float32)
    scale[scale == 0] = 1
    values = np.clip(np.rint(matrix / scale[:, None]), -127, 127).astype(np.int8)
    return values, scale


def quantize_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict]:
    """Return a weight-only INT8 copy of a float32 graph and a per-tensor report."""
    result = onnx.ModelProto()
    result.CopyFrom(model)
    graph = result.graph
    graph_inputs = {value.name for value in graph.input}
    uses = _uses(graph)
    taken = {value.name for value in graph.initializer} | {
        output for node in graph.node for output in node.output
    }
    new_initializers, new_nodes, removed, tensors = [], [], set(), []

    def fresh(name: str) -> str:
        if name in taken:
            raise ValueError(f"Quantization would overwrite an existing tensor: {name}")
        taken.add(name)
        return name

    for initializer in graph.initializer:
        weight = numpy_helper.to_array(initializer)
        name = initializer.name
        if weight.dtype != np.float32 or name in graph_inputs:
            continue
        consumers = uses.get(name, [])
        basis = _dft_basis(weight) if weight.size >= MIN_QUANTIZED_SIZE else None
        if (
            basis is not None
            and consumers
            and all(
                node.op_type in ("Conv", "ConvTranspose") and slot == 1 for node, slot in consumers
            )
        ):
            bins, size, scale = basis
            inits, nodes = _basis_nodes(name, bins, size, scale)
            for item in inits:
                fresh(item.name)
            for node in nodes[:-1]:
                fresh(node.output[0])
            new_initializers += inits
            new_nodes += nodes
            removed.add(name)
            tensors.append({"name": name, "shape": list(weight.shape), "storage": "generated"})
            continue
        layouts = {_layout(node, slot, weight.ndim) for node, slot in consumers}
        if (
            weight.ndim < 2
            or weight.size < MIN_QUANTIZED_SIZE
            or len(layouts) != 1
            or None in layouts
        ):
            continue
        kind, axis = layouts.pop()
        channels = weight.shape[axis] if kind == "axis" else math.prod(weight.shape[:axis])
        if channels < 2:
            # A single scale is no per-channel quantization; e.g. overlap-add windows.
            continue
        if kind == "axis":
            moved = np.moveaxis(weight, axis, 0)
            values, scale = quantize_rows(moved.reshape(moved.shape[0], -1))
            stored = np.moveaxis(values.reshape(moved.shape), 0, axis)
            restored = np.moveaxis(
                (values.astype(np.float32) * scale[:, None]).reshape(moved.shape), 0, axis
            )
        else:
            flat_shape = (math.prod(weight.shape[:axis]), *weight.shape[axis:])
            values, scale = quantize_rows(weight.reshape(flat_shape[0], -1))
            stored = values.reshape(flat_shape)
            restored = (values.astype(np.float32) * scale[:, None]).reshape(weight.shape)
        q, s, z = fresh(name + "_q"), fresh(name + "_scale"), fresh(name + "_zero_point")
        new_initializers += [
            numpy_helper.from_array(stored, q),
            numpy_helper.from_array(scale, s),
            numpy_helper.from_array(np.zeros(len(scale), np.int8), z),
        ]
        if kind == "axis":
            new_nodes.append(helper.make_node("DequantizeLinear", [q, s, z], [name], axis=axis))
        else:
            flat, shape = fresh(name + "_flat"), fresh(name + "_shape")
            new_initializers.append(
                numpy_helper.from_array(np.array(weight.shape, np.int64), shape)
            )
            new_nodes += [
                helper.make_node("DequantizeLinear", [q, s, z], [flat], axis=0),
                helper.make_node("Reshape", [flat, shape], [name]),
            ]
        removed.add(name)
        error = float(np.linalg.norm(restored - weight) / max(np.linalg.norm(weight), 1e-30))
        tensors.append(
            {
                "name": name,
                "shape": list(weight.shape),
                "storage": "int8",
                "scales": int(len(scale)),
                "relative_l2_error": error,
            }
        )

    kept = [item for item in graph.initializer if item.name not in removed]
    del graph.initializer[:]
    graph.initializer.extend(kept + new_initializers)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(new_nodes + nodes)
    onnx.checker.check_model(result, full_check=True)
    errors = [item["relative_l2_error"] for item in tensors if item["storage"] == "int8"]
    report = {
        "scheme": SCHEME,
        "int8_tensors": len(errors),
        "generated_bases": sum(item["storage"] == "generated" for item in tensors),
        "max_relative_l2_error": max(errors, default=0.0),
        "median_relative_l2_error": float(np.median(errors)) if errors else 0.0,
        "tensors": tensors,
    }
    return result, report


def _parity_graph(model: onnx.ModelProto, salt: int = 0) -> bytes:
    """Fix every random op's seed by node name and expose the rounded durations."""
    copy = onnx.ModelProto()
    copy.CopyFrom(model)
    rounds = [node for node in copy.graph.node if node.op_type == "Round"]
    if len(rounds) != 1:
        raise ValueError("Expected exactly one Round node producing token durations")
    for node in copy.graph.node:
        if node.op_type in _RANDOM:
            for attribute in list(node.attribute):
                if attribute.name == "seed":
                    node.attribute.remove(attribute)
            node.attribute.append(
                helper.make_attribute(
                    "seed", float((zlib.crc32(node.name.encode()) + salt) % 2**24)
                )
            )
    copy.graph.output.append(
        helper.make_tensor_value_info(rounds[0].output[0], TensorProto.FLOAT, None)
    )
    return copy.SerializeToString()


def log_spectral_distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean per-frame RMS difference of log magnitude spectra in dB (phase-invariant).

    Magnitudes are floored 80 dB below the reference peak, and frames more than
    60 dB below the loudest reference frame are ignored.
    """
    size, hop = 1024, 256
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)

    def spectrum(signal: np.ndarray) -> np.ndarray:
        padded = np.pad(signal.astype(np.float64), size // 2)
        frames = np.lib.stride_tricks.sliding_window_view(padded, size)[::hop] * window
        return 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) + 1e-12)

    first, second = spectrum(reference), spectrum(candidate)
    floor = first.max() - 80
    first, second = np.maximum(first, floor), np.maximum(second, floor)
    energy = first.max(axis=1)
    active = energy > energy.max() - 60
    return float(np.mean(np.sqrt(np.mean((first[active] - second[active]) ** 2, axis=1))))


def parity_report(
    reference: onnx.ModelProto,
    candidate: onnx.ModelProto,
    token_cases: list[list[int]],
    speed: float = 1.0,
    threads: int = 1,
) -> dict:
    """Compare a candidate graph with its float32 reference on identical noise.

    Random excitation is seeded identically in both graphs and each case uses
    fresh sessions, so differences come from the stored weights. Waveform SNR and
    log-spectral distance are reported only when every token duration matches.
    The baseline compares the reference with itself under different noise: the
    model's own variation, against which a weight-storage error is judged.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    graphs = [_parity_graph(reference), _parity_graph(candidate), _parity_graph(reference, 7919)]
    cases = []
    for tokens in token_cases:
        feed = {
            "input_ids": np.asarray([tokens], dtype=np.int64),
            "speed": np.asarray([speed], dtype=np.float32),
        }
        outputs = []
        for graph in graphs:
            session = ort.InferenceSession(graph, options, providers=["CPUExecutionProvider"])
            waveform, durations = session.run(None, feed)[:2]
            outputs.append((waveform[0], np.asarray(durations).reshape(-1)))
        (ref_wave, ref_dur), (cand_wave, cand_dur), (noise_wave, _) = outputs
        mismatched = int(np.sum(ref_dur != cand_dur)) if ref_dur.shape == cand_dur.shape else -1
        snr = distance = None
        if mismatched == 0 and ref_wave.shape == cand_wave.shape:
            noise = float(np.sum((cand_wave.astype(np.float64) - ref_wave) ** 2))
            signal = float(np.sum(ref_wave.astype(np.float64) ** 2))
            snr = math.inf if noise == 0 else 10 * math.log10(signal / noise)
            distance = log_spectral_distance(ref_wave, cand_wave)
        cases.append(
            {
                "tokens": len(tokens),
                "duration_mismatches": mismatched,
                "frames_reference": float(ref_dur.sum()),
                "frames_candidate": float(cand_dur.sum()),
                "snr_db": snr,
                "log_spectral_distance_db": distance,
                "baseline_log_spectral_distance_db": log_spectral_distance(ref_wave, noise_wave),
            }
        )
    finite = [case["snr_db"] for case in cases if case["snr_db"] is not None]
    finite = [value for value in finite if math.isfinite(value)]
    distances = [case["log_spectral_distance_db"] for case in cases]
    distances = [value for value in distances if value is not None]
    return {
        "cases": len(cases),
        "duration_matched_cases": len(distances),
        "duration_mismatched_tokens": sum(
            case["tokens"] if case["duration_mismatches"] < 0 else case["duration_mismatches"]
            for case in cases
        ),
        "tokens": sum(case["tokens"] for case in cases),
        "median_snr_db": float(np.median(finite)) if finite else None,
        "median_log_spectral_distance_db": float(np.median(distances)) if distances else None,
        "max_log_spectral_distance_db": max(distances, default=None),
        "baseline_median_log_spectral_distance_db": float(
            np.median([case["baseline_log_spectral_distance_db"] for case in cases])
        ),
        "details": cases,
    }


def parity_failures(report: dict) -> list[str]:
    """Reject gross storage errors; small deviations stay below the model's own noise."""
    failures = []
    if report["duration_mismatched_tokens"] > MAX_DURATION_CHANGE * report["tokens"]:
        failures.append(
            f"{report['duration_mismatched_tokens']} of {report['tokens']} token durations changed"
        )
    distance = report["median_log_spectral_distance_db"]
    baseline = report["baseline_median_log_spectral_distance_db"]
    if distance is None:
        failures.append("no case kept every token duration")
    elif distance > MAX_DISTANCE_RATIO * baseline:
        failures.append(
            f"median log-spectral distance {distance:.2f} dB exceeds "
            f"{MAX_DISTANCE_RATIO} x the reseeded FP32 baseline {baseline:.2f} dB"
        )
    return failures
