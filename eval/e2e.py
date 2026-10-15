"""End to end in PyTorch: text-side student -> decoder from token ids, against the same decoder on Kokoro's own
inputs.

  python -m eval.e2e VOICE label=TEXT.pt,DECODER.pt ... [--n 200] [--out FILE]

For each text side and decoder, on the first N held-out rows of VOICE: UTMOS end to end (the text student's
rounded durations, F0, energy and phoneme features; excitation seed 1000 + row), UTMOS of the decoder
teacher-forced on the row (same seed), the paired difference end to end minus teacher-forced, and the length
ratio (predicted frames over the teacher's). The teacher's own audio is scored once per voice.
"""
import argparse

import numpy as np

from training.common import VOICES, device, full
from training.utmos import Utmos

from .common import SEED0, heldout, labelled, load_decoder, load_text, rounded, save_json, synth, teacher_forced
from .stats import paired


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("voice", choices=sorted(VOICES))
    ap.add_argument("models", nargs="+", metavar="label=TEXT.pt,DECODER.pt")
    ap.add_argument("--n", type=int, default=200, help="first N held-out rows")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="JSON file (default out/eval/e2e-VOICE-N.json)")
    a = ap.parse_args(argv)
    dev = device(a.device)
    models = [(label, *paths.split(",")) for label, paths in labelled(a.models)]
    if any(len(m) != 3 for m in models):
        raise SystemExit("expected label=TEXT.pt,DECODER.pt")
    rows = heldout(a.voice, a.n)
    utmos = Utmos(dev)
    teacher = [utmos(full(r)[3]) for r in rows]
    report = {"voice": a.voice, "n": len(rows), "excitation_seeds": f"{SEED0} + row",
              "teacher": {"utmos": round(float(np.mean(teacher)), 4), "per_row": rounded(teacher)}}
    print("teacher", report["teacher"]["utmos"], flush=True)
    for label, text_path, dec_path in models:
        ts, dec = load_text(text_path, dev), load_decoder(dec_path, dev)
        e2e, tf, ratio = [], [], []
        for i, r in enumerate(rows):
            y, n_frames = synth(ts, dec, r["ids"], SEED0 + i, dev)
            e2e.append(utmos(y))
            ratio.append(n_frames / int(r["pred_dur"].sum()))
            tf.append(utmos(teacher_forced(dec, r, SEED0 + i, dev)))
        report[label] = {"text": text_path, "decoder": dec_path,
                         "utmos_end_to_end": round(float(np.mean(e2e)), 4),
                         "utmos_teacher_forced": round(float(np.mean(tf)), 4),
                         "end_to_end_minus_teacher_forced": paired(e2e, tf),
                         "length_ratio": round(float(np.mean(ratio)), 4),
                         "per_row": {"end_to_end": rounded(e2e), "teacher_forced": rounded(tf), "length_ratio": rounded(ratio)}}
        print(label, {k: v for k, v in report[label].items() if k not in ("per_row", "text", "decoder")}, flush=True)
        del ts, dec
    save_json(a.out or f"out/eval/e2e-{a.voice}-{len(rows)}.json", report)


if __name__ == "__main__":
    main()
