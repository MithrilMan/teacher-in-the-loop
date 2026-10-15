"""UTMOS of decoders alone, on Kokoro's own inputs (teacher conditioning).

  python -m eval.decoder VOICE label=DECODER.pt ... [--n 200] [--lock] [--distances] [--out FILE]

Each decoder renders the first N held-out rows of VOICE from the teacher's phoneme features, durations, F0 and
energy for the whole sentence, with excitation seed 1000 + row, so all decoders see the same random draws. The
teacher's own audio of the same rows is scored as well. Reports mean UTMOS with the paper's 95% interval
(1.96 x population SD / sqrt(N)) and writes every per-row score to the JSON file.

--lock also scores every render after Paradee's phase-lock post-filter (phase_lock.py) and reports the paired
gain (paper, "phase-locking post-filter"). --distances adds, per row, the two terms of the stage-one objective
against the teacher's audio (log-mel L1 and multi-resolution STFT, training.common.spec_losses) and the
periodicity of the render and of the teacher's audio.
"""
import argparse
import math

import numpy as np
import torch

from training.common import VOICES, device, full, spec_losses
from training.utmos import Utmos

from .common import SEED0, heldout, labelled, load_decoder, rounded, save_json, teacher_forced
from .phase_lock import lock as phase_lock
from .stats import ci95, paired


def periodicity(x, f0, sr=24000):
    """Mean normalized autocorrelation at the F0 lag over voiced 40 ms frames (F0 at 80 Hz); the lag is searched
    within +-3% of the nominal one because the F0 track is frame-level."""
    x = x.cpu().numpy().astype(np.float64)
    f0 = f0.cpu().numpy()
    vals = []
    for i in range(0, len(f0), 2):
        if f0[i] < 60:
            continue
        lag = int(round(sr / f0[i]))
        c = i * 300
        a = x[c : c + 960]
        b = x[c + lag : c + lag + 960]
        if len(b) < 960 or np.dot(a, a) < 1e-8:
            continue
        best = -1
        for l2 in range(int(lag * 0.97), int(lag * 1.03) + 2):
            b = x[c + l2 : c + l2 + 960]
            if len(b) < 960:
                break
            best = max(best, np.dot(a, b) / math.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-12))
        vals.append(best)
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def score(dec, rows, utmos, dev, distances=False, lock=False):
    """Per-row UTMOS of one decoder; with lock, also after the phase-lock filter; with distances, log-mel L1, MR-STFT
    and periodicity."""
    out = {"utmos": []}
    if lock:
        out["utmos_locked"] = []
        cap = {}
        hook = dec.generator.m_source.register_forward_hook(lambda m, i, o: cap.__setitem__("src", o[0].squeeze().float().cpu()))
    if distances:
        out.update(mel_l1=[], mrstft=[], periodicity=[])
    for i, r in enumerate(rows):
        y = teacher_forced(dec, r, SEED0 + i, dev)
        out["utmos"].append(utmos(y))
        if lock:
            out["utmos_locked"].append(utmos(phase_lock(y.cpu(), cap["src"], frames_f0(r))))
        if distances:
            _, F0, _, au = full(r)
            k = min(len(y), len(au))
            lm, ls = spec_losses(y[None, :k], au[None, :k].to(y.device))
            out["mel_l1"].append(lm.item())
            out["mrstft"].append(ls.item())
            out["periodicity"].append(periodicity(y, F0))
    if lock:
        hook.remove()
    return out


def frames_f0(r):
    return full(r)[1]


def summary(per_row):
    s = {k: round(float(np.mean(v)), 4) for k, v in per_row.items()}
    s["utmos_ci95"] = round(ci95(per_row["utmos"]), 4)
    if "utmos_locked" in per_row:
        s["lock_gain"] = paired(per_row["utmos_locked"], per_row["utmos"])
    s["per_row"] = {k: rounded(v) for k, v in per_row.items()}
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("voice", choices=sorted(VOICES))
    ap.add_argument("decoders", nargs="+", metavar="label=DECODER.pt")
    ap.add_argument("--n", type=int, default=200, help="first N held-out rows")
    ap.add_argument("--lock", action="store_true", help="also score after Paradee's phase-lock post-filter")
    ap.add_argument("--distances", action="store_true", help="also log-mel L1, MR-STFT and periodicity")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="JSON file (default out/eval/decoder-VOICE-N.json)")
    a = ap.parse_args(argv)
    dev = device(a.device)
    decoders = labelled(a.decoders)
    rows = heldout(a.voice, a.n)
    utmos = Utmos(dev)
    teacher = {"utmos": [utmos(full(r)[3]) for r in rows]}
    if a.distances:
        teacher["periodicity"] = [periodicity(full(r)[3], full(r)[1]) for r in rows]
    report = {"voice": a.voice, "n": len(rows), "excitation_seeds": f"{SEED0} + row", "teacher": summary(teacher)}
    print("teacher", {k: v for k, v in report["teacher"].items() if k != "per_row"}, flush=True)
    for label, path in decoders:
        dec = load_decoder(path, dev)
        report[label] = {"path": path, **summary(score(dec, rows, utmos, dev, a.distances, a.lock))}
        print(label, {k: v for k, v in report[label].items() if k not in ("per_row", "path")}, flush=True)
        del dec
    save_json(a.out or f"out/eval/decoder-{a.voice}-{len(rows)}.json", report)


if __name__ == "__main__":
    main()
