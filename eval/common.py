"""Shared pieces of the evaluation: held-out rows, trained students, and renders with fixed excitation seeds."""
import json
import os

import torch

from training.common import frames, gpu_source_stft, run_dir
from training.student_decoder import load_student_decoder

# Held-out row i is rendered with excitation seed SEED0 + i by every decoder and in every condition, so two
# decoders compared on the same rows see the same random draws before their mixer.
SEED0 = 1000
HELDOUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heldout.json")


def heldout(voice, n=200):
    """The first n held-out rows of a voice's teacher corpus (training/gen_teacher.py), with the teacher's audio."""
    path = os.path.join(run_dir(voice), "teacher", "heldout.pt")
    rows = torch.load(path, weights_only=False)
    if len(rows) < n:
        raise ValueError(f"{path} has {len(rows)} rows, {n} requested")
    return rows[:n]


def heldout_ids(language, n=200):
    """Token ids of the first n held-out sentences of a language (eval/heldout.json): the ids of the teacher rows."""
    with open(HELDOUT_JSON, encoding="utf-8") as f:
        items = json.load(f)[language]
    if len(items) < n:
        raise ValueError(f"{HELDOUT_JSON} has {len(items)} {language} sentences, {n} requested")
    return [r["ids"] for r in items[:n]]


def load_decoder(path, dev):
    dec = load_student_decoder(path, dev)
    return gpu_source_stft(dec) if str(dev).startswith("cuda") else dec


def load_text(path, dev):
    """A text-side student (s+mlp, the Paradee text side) from its checkpoint."""
    from training.student import TextStudent

    ts = TextStudent("s+mlp").to(dev).eval()
    ts.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["model"])
    return ts


@torch.no_grad()
def teacher_forced(dec, r, seed, dev):
    """The decoder on the row's own inputs for the whole sentence: Kokoro's phoneme features expanded by its
    durations, its F0 and its energy (teacher conditioning)."""
    asr, F0, N = frames(r)
    torch.manual_seed(seed)
    return dec(asr[None].to(dev), F0[None].to(dev), N[None].to(dev)).squeeze().float()


@torch.no_grad()
def synth(ts, dec, ids, seed, dev):
    """From token ids alone: the text student's durations (rounded, at least one frame), its F0, energy and
    phoneme features, then the decoder. Returns the waveform and the number of frames."""
    from training.student import alignment

    ids = torch.as_tensor(ids, device=dev)[None]
    L = torch.tensor([ids.shape[1]], device=dev)
    dur, _, _, asr_tok = ts(ids, L)
    pred = dur[0].round().clamp(min=1).long()
    aln = alignment(pred)[None]
    _, F0, N, _ = ts(ids, L, aln)
    torch.manual_seed(seed)
    return dec(asr_tok @ aln, F0, N).squeeze().float(), int(pred.sum())


def labelled(args):
    """'label=value' command-line arguments as (label, value) pairs."""
    out = []
    for a in args:
        label, sep, value = a.partition("=")
        if not sep or not label or not value:
            raise SystemExit(f"expected label=value, got {a!r}")
        out.append((label, value))
    return out


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")


def rounded(xs, digits=4):
    return [round(float(x), digits) for x in xs]
