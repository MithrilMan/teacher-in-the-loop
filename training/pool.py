"""Teacher pool: pre-render the decoder's targets for random crops of the training sentences.

  python -m training.pool VOICE [--crops 150000] [--seed 0] [--train-rows N] [--teacher-fp16]

For each 64-frame crop: one excitation drawn from the teacher's source module, the frozen teacher decoder run on
the crop with that excitation, and the output of its iSTFT head stored as (log-magnitude, sin(phase)). The head
rebuilds the teacher's waveform for the crop exactly, so the student's target is a deterministic function of the
inputs it receives, including the excitation. Writes runs/VOICE/pool/{head,src,idx}.npy and meta.json
(about 62 GB for 150,000 crops).
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from .common import Excitation, FixedSource, crop, device, gpu_source_stft, load_rows, load_teacher, run_dir, seed_everything, spec_losses, style

SEG = 64  # frames per crop (1.6 s)
HEAD_FRAMES = SEG * 120 + 1  # iSTFT head frames for a crop: 2 x 60 upsampling, plus the reflection pad


def teacher_head(voice, dev, fp16):
    """The frozen teacher decoder on the device, with a replaceable excitation and a capture of its head output."""
    teacher, pack = load_teacher(voice, "cpu")
    exc = Excitation(teacher, dev)
    dec = gpu_source_stft(teacher.decoder.to(dev).eval().requires_grad_(False))
    if fp16:  # exp/sin and the complex inverse STFT stay in fp32
        inv = dec.generator.stft.inverse

        def inverse(mag, phase):
            with torch.autocast("cuda", enabled=False):
                return inv(mag.float(), phase.float())

        dec.generator.stft.inverse = inverse
    fixed = FixedSource()
    dec.generator.m_source = fixed
    cap = {}
    dec.generator.conv_post.register_forward_hook(lambda m, i, o: cap.__setitem__("head", o))
    return dec, exc, fixed, cap, pack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    ap.add_argument("--crops", type=int, default=150000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-rows", type=int, default=0, help="draw crops from the first N training sentences only")
    ap.add_argument("--teacher-fp16", action="store_true", help="run the teacher under fp16 autocast (the paper's English pool)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = device(a.device)
    seed_everything(a.seed)
    _, train = load_rows(a.voice, a.train_rows)
    dec, exc, fixed, cap, pack = teacher_head(a.voice, dev, a.teacher_fp16)
    out = os.path.join(run_dir(a.voice), "pool")
    os.makedirs(out, exist_ok=True)

    def mm(name, shape, dt):
        return np.lib.format.open_memmap(os.path.join(out, f"{name}.npy"), mode="w+", dtype=dt, shape=shape)

    n = a.crops
    head, src, idx = mm("head", (n, 22, HEAD_FRAMES), np.float16), mm("src", (n, 600 * SEG), np.float16), mm("idx", (n, 2), np.int32)
    t0 = time.time()
    for b0 in range(0, n, a.bs):
        bs = min(a.bs, n - b0)
        rows = [random.randrange(len(train)) for _ in range(bs)]
        crops = [crop(train[i], SEG) for i in rows]
        asr, F0, N = [torch.stack([c[j] for c in crops]).to(dev) for j in range(3)]
        with torch.no_grad():
            fixed.src = exc(F0)
            sty = torch.cat([style(pack, train[i]["ps"])[:, :128] for i in rows]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.teacher_fp16):
                y = dec(asr, F0, N, sty).squeeze(1).float()
            x = cap["head"]
            h = torch.cat([x[:, :11].float(), torch.sin(x[:, 11:].float())], 1).half()
            if b0 == 0:  # the stored head must rebuild the teacher's waveform
                r = dec.generator.stft.inverse(torch.exp(h[:, :11].float()), h[:, 11:].float()).squeeze(1)
                k = min(r.shape[-1], y.shape[-1])
                print(f"pool exactness: max abs {(r[:, :k] - y[:, :k]).abs().max().item():.5f}, "
                      f"mel L1 {spec_losses(r[:, :k], y[:, :k])[0].item():.5f}", flush=True)
        head[b0 : b0 + bs] = h.cpu().numpy()
        src[b0 : b0 + bs] = fixed.src[:, :, 0].half().cpu().numpy()
        idx[b0 : b0 + bs] = np.array([[i, c[3]] for i, c in zip(rows, crops)], np.int32)
        if (b0 // a.bs) % 200 == 0:
            print(f"pool {b0 + bs}/{n} {(time.time() - t0) / (b0 + bs) * 1000:.1f} ms/crop", flush=True)
    for m in (head, src, idx):
        m.flush()
    meta = {"voice": a.voice, "n_train": len(train), "train_rows": a.train_rows, "n_crops": n, "seg": SEG,
            "seed": a.seed, "teacher": "fp16" if a.teacher_fp16 else "fp32"}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    print("pool done", n, flush=True)


if __name__ == "__main__":
    main()
