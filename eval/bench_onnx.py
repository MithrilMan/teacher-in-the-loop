"""Real-time factor of exported graphs on one CPU thread (Paradee's protocol: 8 held-out sentences, from
phonemes, one thread).

  python bench_onnx.py HELDOUT_JSON REPEATS voice:label=path.onnx ...
Per graph: one warm-up pass over the 8 sentences, then REPEATS timed passes; reports seconds of audio per
second of compute (x real time) for the median and the best pass, and the audio length.
"""
import json, platform, statistics, sys, time

import numpy as np
import onnxruntime as ort

LANG = {"af_heart": "en", "if_sara": "it", "im_nicola": "it"}
held = json.load(open(sys.argv[1], encoding="utf-8"))
repeats = int(sys.argv[2])
report = {"onnxruntime": ort.__version__, "machine": platform.processor(), "threads": 1, "results": {}}
for spec in sys.argv[3:]:
    key, path = spec.split("=", 1)
    voice, label = key.split(":")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    feeds = [{"input_ids": np.asarray(x, dtype=np.int64)[None], "speed": np.ones(1, np.float32)} for x in [r["ids"] for r in held[LANG[voice]][:8]]]

    def one_pass():
        n = 0
        t = time.perf_counter()
        for f in feeds:
            n += sess.run(["waveform"], f)[0].shape[-1]
        return time.perf_counter() - t, n / 24000

    one_pass()
    passes = [one_pass() for _ in range(repeats)]
    audio = passes[0][1]
    xs = [audio / t for t, _ in passes]
    report["results"][f"{voice}:{label}"] = {"audio_s": round(audio, 2), "x_realtime_median": round(statistics.median(xs), 1),
                                             "x_realtime_best": round(max(xs), 1), "passes_s": [round(t, 3) for t, _ in passes]}
    print(f"{voice}:{label}", json.dumps(report["results"][f"{voice}:{label}"]), flush=True)
print(json.dumps(report))
