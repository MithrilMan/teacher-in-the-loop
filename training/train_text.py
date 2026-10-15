# Derived from Paradee's training/scripts/train_text.py (Apache-2.0); see NOTICE.
"""Text side: Paradee's text student (s+mlp, 4.23M parameters), trained on the teacher's intermediates with the
upstream recipe and losses. It predicts durations, F0, N and the decoder's phoneme features from token ids.

  python -m training.train_text VOICE [--steps 8000]

Writes runs/VOICE/text/{last.pt,final.pt,log.jsonl}; rerunning the command resumes from last.pt.
"""
import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import device, load_rows, log_json, run_dir, save, seed_everything
from .student import TextStudent, alignment


def collate(batch):
    B = len(batch)
    T = max(len(r["ids"]) for r in batch)
    Fm = max(len(r["F0"]) for r in batch) // 2
    ids = torch.zeros(B, T, dtype=torch.long)
    L = torch.tensor([len(r["ids"]) for r in batch])
    dur, ten, aln = torch.zeros(B, T), torch.zeros(B, 512, T), torch.zeros(B, T, Fm)
    F0, N, fm = torch.zeros(B, 2 * Fm), torch.zeros(B, 2 * Fm), torch.zeros(B, 2 * Fm, dtype=torch.bool)
    for i, r in enumerate(batch):
        t, f = len(r["ids"]), len(r["F0"]) // 2
        ids[i, :t], dur[i, :t], ten[i, :, :t] = r["ids"], r["dur"], r["t_en"].float()
        aln[i, :t, :f] = alignment(r["pred_dur"], f)
        F0[i, : 2 * f], N[i, : 2 * f], fm[i, : 2 * f] = r["F0"], r["N"], True
    tm = torch.arange(T)[None] < L[:, None]
    return ids, L, dur, ten, aln, F0, N, tm, fm


def losses(model, batch, dev):
    """Teacher-forced: L1 on log durations, F0 (/100), N and the phoneme features (/512)."""
    ids, L, dur, ten, aln, F0, N, tm, fm = [x.to(dev) for x in batch]
    pd, pF0, pN, pasr = model(ids, L, aln)
    l_dur = (F.l1_loss(torch.log(pd.clamp_min(1e-3)), torch.log(dur.clamp_min(1e-3)), reduction="none") * tm).sum() / tm.sum()
    l_f0 = (F.l1_loss(pF0, F0, reduction="none") * fm).sum() / fm.sum() / 100
    l_n = (F.l1_loss(pN, N, reduction="none") * fm).sum() / fm.sum()
    l_asr = (F.l1_loss(pasr, ten, reduction="none") * tm[:, None]).sum() / tm.sum() / 512
    return l_dur, l_f0, l_n, l_asr


@torch.no_grad()
def evaluate(model, held, dev):
    """Held-out losses, and the free-running duration error in frames."""
    model.eval()
    acc = np.mean([[x.item() for x in losses(model, collate(held[i : i + 32]), dev)] for i in range(0, len(held), 32)], 0)
    mae = []
    for r in held:
        ids = r["ids"][None].to(dev)
        pd = model(ids, torch.tensor([ids.shape[1]], device=dev))[0][0]
        mae.append((torch.round(pd).clamp(min=1).long().cpu() - r["pred_dur"]).abs().float().mean().item())
    model.train()
    return dict(zip(("dur", "f0", "n", "asr"), np.round(acc, 4).tolist()), dur_mae_frames=round(float(np.mean(mae)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = device(a.device)
    seed_everything(a.seed)
    out = os.path.join(run_dir(a.voice), "text")
    os.makedirs(out, exist_ok=True)
    held, train = load_rows(a.voice)
    print(f"{len(train)} training / {len(held)} held-out sentences", flush=True)
    model = TextStudent("s+mlp").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    step, last = 0, os.path.join(out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        random.setstate(ck["random"])
        print(f"resumed at step {step}", flush=True)
    print(f"text student: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)
    model.train()
    t0, t_step = time.time(), step
    while step < a.steps:
        batch = collate(random.sample(train, a.bs))
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        ls = losses(model, batch, dev)
        loss = sum(ls)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} " + " ".join(f"{n} {v.item():.4f}" for n, v in zip(("dur", "f0", "n", "asr"), ls))
                  + f" {(time.time() - t0) / (step - t_step):.3f} s/step", flush=True)
        if step % 500 == 0 or step == a.steps:
            save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "random": random.getstate()}, last)
        if step % 2000 == 0 or step == a.steps:
            ev = {"step": step, **evaluate(model, held, dev)}
            log_json(os.path.join(out, "log.jsonl"), ev)
            print("EVAL", ev, flush=True)
    save({"model": model.state_dict(), "voice": a.voice, "args": vars(a)}, os.path.join(out, "final.pt"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
