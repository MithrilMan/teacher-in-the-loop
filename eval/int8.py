"""End to end on exported graphs with ONNX Runtime: the float32 graph against its int8 copy (or any other variant
of it) on identical inputs and identical random excitation.

  python -m eval.int8 VOICE FP32.onnx [label=OTHER.onnx ...] [--variants text_only,decoder_only,keep3]
                      [--n 100] [--threads 4] [--out FILE]

Every graph synthesizes the first N held-out sentences of the voice's language from their token ids
(eval/heldout.json) at speed 1.0, on the CPU; UTMOS scores each waveform. Each other graph is compared with the
float32 graph row by row (other minus float32: mean, standard error, paired t-test, sign test) and its size is
reported. --variants also builds partial int8 copies of the float32 graph with training/quantize.py, some weights
left in float32: text_only (int8 text side, float32 decoder), decoder_only (the reverse) and keep3 (all int8
except the decoder's generator.conv_post, generator.ups.1 and generator.resblocks.4).

The harmonic source of a graph draws from ONNX Runtime's random ops, which ignore numpy and torch seeds; without a
seed attribute they are seeded from a process-wide value that changes from one run of the program to the next.
Here every random op is given a seed computed from its node name, so the float32 graph and an int8 copy with the
same node names draw the same excitation, and the result is the same in every run. A random op keeps drawing
across calls of the same session, so the sentences are run in order through one session per graph: where a graph
predicts a different number of frames for a sentence, the draws of the following sentences shift as well.
Needs onnx, onnxruntime and, for UTMOS, torch.
"""
import argparse
import zlib

import numpy as np
import torch

from training.common import VOICES, device
from training.utmos import Utmos

from .common import heldout_ids, labelled, rounded, save_json
from .stats import paired

RANDOM_OPS = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike"}
# Weights kept in float32 by each partial int8 copy. training/export.py names the text side's weights ts.* and the
# decoder's ds.*; weights the exporter renames (onnx::*) count as text side.
VARIANTS = {
    "text_only": lambda name: name.startswith("ds."),
    "decoder_only": lambda name: not name.startswith("ds."),
    "keep3": lambda name: name.startswith(("ds.generator.conv_post.", "ds.generator.ups.1.", "ds.generator.resblocks.4.")),
}


def seeded(m):
    """Give every random op of the graph (an onnx.ModelProto, changed in place) a seed computed from its node name;
    returns the names of those ops."""
    from onnx import helper

    names = []
    for node in m.graph.node:
        if node.op_type in RANDOM_OPS:
            for attr in list(node.attribute):
                if attr.name == "seed":
                    node.attribute.remove(attr)
            node.attribute.append(helper.make_attribute("seed", float(zlib.crc32(node.name.encode()) % 2**24)))
            names.append(node.name)
    return sorted(names)


def quantize_keep(g, keep):
    """An int8 copy of the float32 graph g (training/quantize.py) with the weights for which keep(name) holds left
    in float32: they are declared as graph inputs while quantizing, which the quantizer leaves alone."""
    import onnx
    from onnx import helper

    from training.quantize import quantize_model

    m = onnx.ModelProto()
    m.CopyFrom(g)
    kept = [i for i in m.graph.initializer if keep(i.name)]
    for i in kept:
        m.graph.input.append(helper.make_tensor_value_info(i.name, i.data_type, list(i.dims)))
    q, _ = quantize_model(m)
    names = {i.name for i in kept}
    inputs = [x for x in q.graph.input if x.name not in names]
    del q.graph.input[:]
    q.graph.input.extend(inputs)
    return q


def render(model_bytes, ids, threads):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    sess = ort.InferenceSession(model_bytes, so, providers=["CPUExecutionProvider"])
    speed = np.ones(1, np.float32)
    return [sess.run(["waveform"], {"input_ids": np.asarray(x, dtype=np.int64)[None], "speed": speed})[0][0] for x in ids]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("voice", choices=sorted(VOICES))
    ap.add_argument("fp32", metavar="FP32.onnx", help="the reference graph")
    ap.add_argument("others", nargs="*", metavar="label=OTHER.onnx")
    ap.add_argument("--variants", default="", help=f"comma-separated partial int8 copies to build: {', '.join(VARIANTS)}")
    ap.add_argument("--n", type=int, default=100, help="first N held-out sentences")
    ap.add_argument("--threads", type=int, default=4, help="ONNX Runtime intra-op threads")
    ap.add_argument("--device", default=None, help="device of the UTMOS model")
    ap.add_argument("--out", default=None, help="JSON file (default out/eval/int8-VOICE-N.json)")
    a = ap.parse_args(argv)
    import onnx

    others = labelled(a.others)
    variants = [v for v in a.variants.split(",") if v]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; available: {', '.join(VARIANTS)}")
    labels = [label for label, _ in others] + [f"int8_{v}" for v in variants]
    if "fp32" in labels or len(set(labels)) != len(labels):
        raise SystemExit("labels must be unique and fp32 is taken by the reference graph")
    ids = heldout_ids(VOICES[a.voice], a.n)
    utmos = Utmos(device(a.device))
    g = onnx.load(a.fp32)

    def evaluate(m):
        size = m.ByteSize()  # as stored, before the seeds are added
        names = seeded(m)
        waves = render(m.SerializeToString(), ids, a.threads)
        return names, size, [utmos(torch.from_numpy(w)) for w in waves], [len(w) for w in waves]

    ref_names, size, ref, ref_len = evaluate(onnx.load(a.fp32))
    report = {"voice": a.voice, "n": len(ids), "speed": 1.0, "random_ops": ref_names,
              "fp32": {"path": a.fp32, "mb": round(size / 1e6, 2), "utmos": round(float(np.mean(ref)), 4),
                       "per_row": rounded(ref), "samples": ref_len}}
    print("fp32", report["fp32"]["utmos"], flush=True)
    graphs = [(label, path, lambda p=path: onnx.load(p)) for label, path in others]
    graphs += [(f"int8_{v}", f"{a.fp32} ({v})", lambda v=v: quantize_keep(g, VARIANTS[v])) for v in variants]
    for label, source, load in graphs:
        names, size, scores, lengths = evaluate(load())
        if names != ref_names:
            raise ValueError(f"{source}: its random ops are not named as in the float32 graph, so the excitation would differ")
        report[label] = {"path": source, "mb": round(size / 1e6, 2), "utmos": round(float(np.mean(scores)), 4),
                         "minus_fp32": paired(scores, ref),
                         "rows_with_other_length": sum(x != y for x, y in zip(lengths, ref_len)),
                         "per_row": rounded(scores), "samples": lengths}
        print(label, {k: v for k, v in report[label].items() if k not in ("per_row", "samples", "path")}, flush=True)
    save_json(a.out or f"out/eval/int8-{a.voice}-{len(ids)}.json", report)


if __name__ == "__main__":
    main()
