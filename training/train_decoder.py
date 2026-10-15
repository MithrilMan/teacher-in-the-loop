# Derived from Paradee's training/scripts/train_decoder.py (Apache-2.0); see NOTICE.
"""Decoder training on the teacher pool: stage one, the warm restart, and the two adversarial stages.

  python -m training.train_decoder VOICE NAME --init scratch|CHECKPOINT [options]

Every step draws crops from the pool (pool.py), rebuilds each crop's target from the stored teacher head, and
feeds the student the very excitation the teacher was run with. The loss is Paradee's spectral objective
(log-mel L1 + multi-resolution STFT) plus L1 on the teacher's iSTFT head: its log-magnitude and sin(phase).
With --gan, Paradee's critics are added (LSGAN + 2 x feature matching) and every reconstruction term is scaled by
--mel-weight. The student's own source module is the teacher's, copied in and frozen; it is what the decoder
uses at inference. Writes runs/VOICE/NAME/{last.pt,final.pt,log.jsonl}; rerunning the command resumes from last.pt.
training/recipe.py runs the stages in order with the paper's settings.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import FixedSource, crop, device, full, gpu_source_stft, load_rows, load_teacher, log_json, run_dir, save, seed_everything, spec_losses
from .student_decoder import StudentDecoder


@torch.no_grad()
def evaluate(model, rows, utmos=None, seed0=1000):
    """The decoder alone on held-out sentences (the teacher's features, excitation seed seed0 + row): distance to
    the teacher's waveform, and UTMOS when a predictor is given."""
    model.eval()
    acc = {"mel": [], "stft": [], "utmos": []}
    dev = next(model.parameters()).device
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        for i, r in enumerate(rows):
            asr, F0, N, au = full(r)
            torch.manual_seed(seed0 + i)
            y = model(asr[None].to(dev), F0[None].to(dev), N[None].to(dev)).squeeze().float()
            k = min(len(y), len(au))
            lm, ls = spec_losses(y[None, :k], au[None, :k].to(dev))
            acc["mel"].append(lm.item())
            acc["stft"].append(ls.item())
            if utmos:
                acc["utmos"].append(utmos(y))
    model.train()
    return {k: round(float(np.mean(v)), 4) for k, v in acc.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    ap.add_argument("name", help="run name: the output goes to runs/VOICE/NAME")
    ap.add_argument("--init", default="scratch", help="'scratch' or a decoder checkpoint (a new optimizer either way)")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=2e-4, help="peak learning rate (generator; the critics keep it constant)")
    ap.add_argument("--warmup", type=int, default=1000, help="linear warmup steps, then a cosine to zero at --steps")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--head-weight", type=float, default=1.0, help="L1 on the teacher head's log-magnitude")
    ap.add_argument("--phase-weight", type=float, default=1.0, help="L1 on the teacher head's sin(phase)")
    ap.add_argument("--gan", action="store_true", help="add Paradee's MPD + MRD critics")
    ap.add_argument("--mel-weight", type=float, default=45.0, help="weight of the reconstruction terms under --gan")
    ap.add_argument("--train-rows", type=int, default=0, help="must match the pool's --train-rows")
    ap.add_argument("--every", type=int, default=2500, help="evaluate and save every this many steps")
    ap.add_argument("--eval-rows", type=int, default=16)
    ap.add_argument("--utmos", action="store_true", help="score the evaluation sentences with UTMOS")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = device(a.device)
    seed_everything(a.seed)
    rng = np.random.default_rng(a.seed)
    model = StudentDecoder("A")
    out = os.path.join(run_dir(a.voice), a.name)
    os.makedirs(out, exist_ok=True)

    held, train = load_rows(a.voice, a.train_rows)
    rows = held[: a.eval_rows]
    pdir = os.path.join(run_dir(a.voice), "pool")
    meta = json.load(open(os.path.join(pdir, "meta.json")))
    if meta["voice"] != a.voice or meta["n_train"] != len(train):
        raise ValueError(f"the pool was drawn from {meta['n_train']} sentences of {meta['voice']}, not {len(train)} of {a.voice}")
    pool = {k: np.load(os.path.join(pdir, f"{k}.npy"), mmap_mode="r") for k in ("head", "src", "idx")}
    seg = meta.get("seg", 64)
    print(f"{len(train)} training sentences, pool of {meta['n_crops']} crops", flush=True)

    teacher, _ = load_teacher(a.voice, "cpu")
    if a.init != "scratch":
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"initialized from {a.init}", flush=True)
    model.generator.m_source.load_state_dict(teacher.decoder.generator.m_source.state_dict())
    model.generator.m_source.requires_grad_(False)
    istft = teacher.decoder.generator.stft  # the head's inverse STFT (n_fft 20, hop 5), parameter-free
    del teacher
    model = gpu_source_stft(model.to(dev))
    own_source, fixed = model.generator.m_source, FixedSource()
    cap = {}
    model.generator.conv_post.register_forward_hook(lambda m, i, o: cap.__setitem__("head", o))
    print(f"decoder: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.8, 0.99), weight_decay=0.01)
    if a.gan:
        from .discriminators import Discriminators, d_loss, g_loss

        D = Discriminators().to(dev)
        optD = torch.optim.AdamW(D.parameters(), lr=a.lr, betas=(0.8, 0.99), weight_decay=0.01)
    sched = lambda s: min(1, s / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    utmos = None
    if a.utmos:
        from .utmos import Utmos

        utmos = Utmos(dev)

    step, last = 0, os.path.join(out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if a.gan:
            D.load_state_dict(ck["D"])
            optD.load_state_dict(ck["optD"])
        step, rng = ck["step"], ck["rng"]
        print(f"resumed at step {step}", flush=True)
    else:
        log_json(os.path.join(out, "log.jsonl"), {"step": 0, **evaluate(model, rows, utmos)})

    model.train()
    t0, t_step, acc = time.time(), step, []
    while step < a.steps:
        step += 1
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        pick = np.sort(rng.choice(len(pool["idx"]), a.bs, replace=False))
        crops = [crop(train[i], seg, start=int(st)) for i, st in pool["idx"][pick]]
        asr, F0, N = [torch.stack([c[j] for c in crops]).to(dev) for j in range(3)]
        fixed.src = torch.from_numpy(pool["src"][pick].astype(np.float32))[:, :, None].to(dev)
        ht = torch.from_numpy(pool["head"][pick].astype(np.float32)).to(dev)
        with torch.no_grad():
            au = istft.inverse(torch.exp(ht[:, :11]), ht[:, 11:]).squeeze(1)

        model.generator.m_source = fixed
        y = model(asr, F0, N).squeeze(1)
        model.generator.m_source = own_source
        k = min(y.shape[-1], au.shape[-1])
        y, au = y[:, :k], au[:, :k]
        lm, ls = spec_losses(y, au)
        hs = cap["head"]
        k2 = min(hs.shape[-1], ht.shape[-1])
        l_head = F.l1_loss(hs[:, :11, :k2], ht[:, :11, :k2])
        l_phase = F.l1_loss(torch.sin(hs[:, 11:, :k2]), ht[:, 11:, :k2])
        rec = lm + ls + a.head_weight * l_head + a.phase_weight * l_phase
        if a.gan:
            ld = d_loss(D, au, y)
            optD.zero_grad(set_to_none=True)
            ld.backward()
            optD.step()
            adv, fm = g_loss(D, au, y)
            loss = a.mel_weight * rec + adv + 2.0 * fm
        else:
            loss = rec
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        acc.append(torch.stack([lm, ls, l_head, l_phase] + ([ld, adv, fm] if a.gan else [])).detach())
        if step % 100 == 0:
            names = ["mel", "stft", "head", "phase"] + (["d", "adv", "fm"] if a.gan else [])
            m = torch.stack(acc).mean(0).tolist()
            acc = []
            print(f"step {step} " + " ".join(f"{n} {v:.4f}" for n, v in zip(names, m))
                  + f" {(time.time() - t0) / (step - t_step):.3f} s/step", flush=True)
        if step % a.every == 0 or step == a.steps:
            ev = {"step": step, **evaluate(model, rows, utmos)}
            log_json(os.path.join(out, "log.jsonl"), ev)
            print("EVAL", ev, flush=True)
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "rng": rng}
            if a.gan:
                ck.update(D=D.state_dict(), optD=optD.state_dict())
            save(ck, last)
    save({"model": model.state_dict(), "voice": a.voice, "args": vars(a)}, os.path.join(out, "final.pt"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
