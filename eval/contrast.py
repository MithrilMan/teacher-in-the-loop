"""Harmonic contrast per band and harmonics-to-noise ratio of voiced speech, for the teacher's audio and for
decoders rendering from the teacher's own inputs.

  python -m eval.contrast VOICE label=DECODER.pt ... [--n 40] [--out FILE]

Each decoder renders the first N held-out rows of VOICE teacher-forced (excitation seed 1000 + row), so its
pitch is exact by construction. On every waveform, F0 is estimated with YIN on 300-sample hops (Kokoro's F0
rate). For each voiced hop whose 1024-sample frame lies within 25 dB of the waveform's 95th-percentile frame
level, a 40 ms Hann window centred on the hop is transformed with a 4096-point FFT; for every harmonic k in a band,
the peak is the highest level within +-10% of F0 around k F0 and the valley the mean level within +-10% of F0
around (k + 1/2) F0. The hop's contrast is mean(peaks) - mean(valleys) in dB. Bands 0-1, 1-2, 2-4 and 4-6 kHz.
HNR: normalized autocorrelation r at the YIN pitch period over 40 ms, 10 log10(r / (1 - r)), over all voiced hops.
Both are averaged over hops, then over rows.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from training.common import SR, VOICES, device, full

from .common import SEED0, heldout, labelled, load_decoder, rounded, save_json, teacher_forced

YIN_HOP = 300
NFFT, HOP = 1024, 256  # frame levels
CN = 4096  # contrast spectrum
CWIN = np.hanning(960)
BANDS = {"0-1k": (0, 1000), "1-2k": (1000, 2000), "2-4k": (2000, 4000), "4-6k": (4000, 6000)}


def yin(x, win=1200, fmin=50.0, fmax=600.0, thr=0.15):
    """F0 per 300-sample hop (frame centred on i * 300), 0 where unvoiced. x: a waveform tensor."""
    maxlag, minlag = int(SR / fmin), int(SR / fmax)
    n = len(x) // YIN_HOP
    xp = F.pad(x[None, None], (win // 2, win // 2 + maxlag), mode="reflect")[0, 0]
    fr = xp.unfold(0, win + maxlag, YIN_HOP)[:n]
    base = fr[:, :win]
    d = torch.stack([((base - fr[:, t : t + win]) ** 2).sum(1) for t in range(maxlag + 1)], 1)
    cm = d[:, 1:] * torch.arange(1, maxlag + 1, device=x.device) / d[:, 1:].cumsum(1).clamp_min(1e-12)
    cm = torch.cat([torch.ones_like(cm[:, :1]), cm], 1)
    cm[:, :minlag] = 2.0
    below = cm < thr
    has = below.any(1)
    tau = torch.where(has, below.float().argmax(1), cm.argmin(1))
    for _ in range(40):  # walk down to the local minimum
        nxt = (tau + 1).clamp(max=maxlag)
        step = cm.gather(1, nxt[:, None])[:, 0] < cm.gather(1, tau[:, None])[:, 0]
        tau = torch.where(step, nxt, tau)
    t0 = tau.clamp(1, maxlag - 1)
    a, b, c = (cm.gather(1, (t0 + k)[:, None])[:, 0] for k in (-1, 0, 1))
    den = a - 2 * b + c
    shift = (0.5 * (a - c) / den.where(den.abs() > 1e-9, torch.ones_like(a))).clamp(-1, 1)
    f0 = SR / (t0.float() + shift)
    voiced = has & (cm.gather(1, tau[:, None])[:, 0] < thr) & (base.pow(2).mean(1) > 1e-6)
    return torch.where(voiced, f0, torch.zeros_like(f0))


def frame_levels(x):
    """Level (dB) of 1024-sample frames, hop 256."""
    n = max(1, (len(x) - NFFT) // HOP + 1)
    idx = np.arange(NFFT)[None] + HOP * np.arange(n)[:, None]
    return 10 * np.log10((np.pad(x, (0, NFFT))[idx] ** 2).mean(1) + 1e-12)


def contrast(x, f0, loud):
    """Mean harmonic contrast (dB) per band over the voiced hops for which loud(hop) holds."""
    acc = {k: [] for k in BANDS}
    for i, f in enumerate(f0):
        if f <= 0 or not loud(i):
            continue
        a = i * YIN_HOP - 480
        if a < 0 or a + 960 > len(x):
            continue
        p = 10 * np.log10(np.abs(np.fft.rfft(x[a : a + 960] * CWIN, CN)) ** 2 + 1e-12)
        w = max(1, int(0.1 * f / (SR / CN)))
        for name, (lo, hi) in BANDS.items():
            ks = np.arange(max(1, int(np.ceil(lo / f))), int(hi / f) + 1)
            if not len(ks):
                continue
            peak = [p[max(0, int(round(k * f * CN / SR)) - w) : int(round(k * f * CN / SR)) + w + 1].max() for k in ks]
            valley = [p[max(0, int(round((k + 0.5) * f * CN / SR)) - w) : int(round((k + 0.5) * f * CN / SR)) + w + 1].mean() for k in ks]
            acc[name].append(float(np.mean(peak) - np.mean(valley)))
    return {f"contrast_{k}_db": float(np.mean(v)) if v else np.nan for k, v in acc.items()}


def hnr(x, f0):
    """Mean HNR (dB) over voiced hops: normalized autocorrelation at the pitch period over 40 ms."""
    out = []
    for i, f in enumerate(f0):
        if f <= 0:
            continue
        T = int(round(SR / f))
        a = i * YIN_HOP - 480
        if a < 0 or a + 960 + T > len(x):
            continue
        u, v = x[a : a + 960], x[a + T : a + 960 + T]
        r = float((u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum() + 1e-12))
        r = min(max(r, 1e-3), 0.999)
        out.append(10 * np.log10(r / (1 - r)))
    return float(np.mean(out)) if out else np.nan


def measure(x, dev):
    """Contrast per band and HNR of one waveform (numpy, 24 kHz)."""
    lvl = frame_levels(x)
    top = np.percentile(lvl, 95)
    f0 = yin(torch.from_numpy(x).float().to(dev)).cpu().numpy()
    loud = lambda i: lvl[min(len(lvl) - 1, max(0, (i * YIN_HOP - NFFT // 2) // HOP))] > top - 25  # noqa: E731
    return {**contrast(x, f0, loud), "hnr_db": hnr(x, f0)}


def summary(per_row):
    keys = sorted(per_row[0])
    return {**{k: round(float(np.nanmean([m[k] for m in per_row])), 2) for k in keys},
            "per_row": {k: rounded([m[k] for m in per_row], 2) for k in keys}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("voice", choices=sorted(VOICES))
    ap.add_argument("decoders", nargs="*", metavar="label=DECODER.pt")
    ap.add_argument("--n", type=int, default=40, help="first N held-out rows")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="JSON file (default out/eval/contrast-VOICE-N.json)")
    a = ap.parse_args(argv)
    dev = device(a.device)
    decoders = labelled(a.decoders)
    rows = heldout(a.voice, a.n)
    report = {"voice": a.voice, "n": len(rows), "excitation_seeds": f"{SEED0} + row",
              "teacher": summary([measure(full(r)[3].numpy(), dev) for r in rows])}
    print("teacher", {k: v for k, v in report["teacher"].items() if k != "per_row"}, flush=True)
    for label, path in decoders:
        dec = load_decoder(path, dev)
        ms = [measure(teacher_forced(dec, r, SEED0 + i, dev).cpu().numpy(), dev) for i, r in enumerate(rows)]
        report[label] = {"path": path, **summary(ms)}
        print(label, {k: v for k, v in report[label].items() if k not in ("per_row", "path")}, flush=True)
        del dec
    save_json(a.out or f"out/eval/contrast-{a.voice}-{len(rows)}.json", report)


if __name__ == "__main__":
    main()
