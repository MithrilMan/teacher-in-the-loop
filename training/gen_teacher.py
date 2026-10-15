# Derived from Paradee's training/scripts/gen_teacher.py (Apache-2.0); see NOTICE.
"""Teacher corpus: run Kokoro over a voice's sentences and store every intermediate the students learn from.

  python -m training.gen_teacher VOICE [--device cuda] [--shard 500]

Per sentence: phonemes and token ids, durations (before and after rounding), the phoneme features t_en (fp16),
F0 and N, and a seed. The 200 held-out sentences also keep the teacher's waveform, which only evaluation uses:
the decoder's targets come from the teacher itself during training (pool.py). Sentences whose phonemes are empty
or longer than 400 symbols are skipped, as in Paradee. Writes runs/VOICE/teacher/heldout.pt and shard_NNNN.pt;
existing shards are kept, so an interrupted run resumes.
"""
import argparse
import json
import os
import time

import torch

from titl.phonemes import Phonemizer, encode, load_vocab

from .common import SENTENCES, VOICES, cpu_source_stft, device, load_teacher, run_dir, save, style

ap = argparse.ArgumentParser()
ap.add_argument("voice", choices=sorted(VOICES))
ap.add_argument("--device", default=None)
ap.add_argument("--shard", type=int, default=500, help="training sentences per shard file")
ap.add_argument("--seed", type=int, default=1000, help="row seeds are seed + index (held-out first)")
ap.add_argument("--max-train", type=int, default=0, help="render only the first N training sentences (for a quick test)")
A = ap.parse_args()
DEV = device(A.device)
lang = VOICES[A.voice]
sents = json.load(open(SENTENCES[lang], encoding="utf-8"))
phonemize, vocab = Phonemizer(lang), load_vocab()
accept = lambda ps: bool(ps) and len(ps) <= 400
if lang == "en":  # Paradee's list: the first 200 accepted sentences are held out
    pos, held_texts = 0, []
    while len(held_texts) < 200:
        if accept(phonemize(sents[pos])):
            held_texts.append(sents[pos])
        pos += 1
    train_texts = sents[pos:]
else:
    held_texts, train_texts = [r["text"] for r in sents["heldout"]], [r["text"] for r in sents["train"]]
teacher, pack = load_teacher(A.voice, DEV)
cpu_source_stft(teacher.decoder)


@torch.inference_mode()
def run(text, seed, with_audio):
    ps = phonemize(text)
    if not accept(ps):
        return None
    m = teacher
    ids = torch.tensor([encode(ps, vocab)], device=DEV)
    T = ids.shape[1]
    ref = style(pack, ps).to(DEV)
    L = torch.full((1,), T, device=DEV)
    mask = torch.zeros(1, T, dtype=torch.bool, device=DEV)
    d_en = m.bert_encoder(m.bert(ids, attention_mask=(~mask).int())).transpose(-1, -2)
    s = ref[:, 128:]
    d = m.predictor.text_encoder(d_en, s, L, mask)
    x, _ = m.predictor.lstm(d)
    dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1)
    pred_dur = torch.round(dur).clamp(min=1).long().squeeze(0)
    idx = torch.repeat_interleave(torch.arange(T, device=DEV), pred_dur)
    aln = torch.zeros(T, len(idx), device=DEV)
    aln[idx, torch.arange(len(idx), device=DEV)] = 1
    F0, N = m.predictor.F0Ntrain(d.transpose(-1, -2) @ aln[None], s)
    t_en = m.text_encoder(ids, L, mask).half().float()  # stored in fp16: the audio is rendered from the stored value
    row = dict(text=text, ps=ps, seed=seed, ids=ids[0].cpu(), dur=dur[0].cpu(), pred_dur=pred_dur.cpu(),
               t_en=t_en[0].half().cpu(), F0=F0[0].cpu(), N=N[0].cpu())
    if with_audio:
        torch.manual_seed(seed)
        audio = m.decoder(t_en @ aln[None], F0, N, ref[:, :128]).flatten()
        assert audio.numel() == 600 * len(idx), "unexpected teacher frame/audio alignment"
        row["audio"] = (audio.cpu().clamp(-1, 1) * 32767).short()
    return row


out = os.path.join(run_dir(A.voice), "teacher")
os.makedirs(out, exist_ok=True)
t0, secs = time.time(), 0.0
held_path = os.path.join(out, "heldout.pt")
if not os.path.exists(held_path):
    held = [run(t, A.seed + i, True) for i, t in enumerate(held_texts)]
    if None in held:
        raise ValueError("a held-out sentence was rejected")
    save(held, held_path)
    print(f"{held_path}: {len(held)} sentences, {time.time() - t0:.0f} s", flush=True)

if A.max_train:
    train_texts = train_texts[: A.max_train]
base = A.seed + 200
for k, start in enumerate(range(0, len(train_texts), A.shard)):
    path = os.path.join(out, f"shard_{k:04d}.pt")
    if os.path.exists(path):
        continue
    rows = [r for i, t in enumerate(train_texts[start : start + A.shard]) if (r := run(t, base + start + i, False))]
    save(rows, path)
    secs += sum(600 * int(r["pred_dur"].sum()) for r in rows) / 24000
    print(f"{path}: {len(rows)} sentences, {secs / 3600:.2f} h of speech, {time.time() - t0:.0f} s", flush=True)
print("done")
