"""Intelligibility and speaker similarity: Whisper word error rate and WavLM x-vector similarity, for the teacher's
audio and for each text side and decoder, teacher-forced and end to end.

  python -m eval.wer_similarity VOICE label=TEXT.pt,DECODER.pt ... [--n 200] [--out FILE]

On the first N held-out rows of VOICE, three conditions: the teacher's own audio; the decoder on the teacher's
inputs (teacher-forced, excitation seed 1000 + row); the text student and decoder from token ids (end to end,
same seed).
- WER: Whisper large-v3-turbo (fp16 on CUDA, language forced, batch 16) on audio resampled to 16 kHz; sentence
  text and transcript are lower-cased with accents and punctuation removed; word-level edit distance
  (substitutions + deletions + insertions) summed over rows / reference words. Also the per-row WER difference
  from the teacher (paired mean and standard error).
- Speaker similarity: cosine of L2-normalized WavLM-base-plus-sv x-vectors, against the teacher's audio of the
  same row (mean and standard error) and against the normalized mean x-vector of the teacher on the first 40
  training rows of the voice. Training rows are stored without audio, so the teacher renders them first, as
  training/gen_teacher.py renders the held-out rows.
Needs transformers and scipy; the models come from the Hugging Face hub.
"""
import argparse
import re
import unicodedata

import numpy as np
import torch

from training.common import VOICES, cpu_source_stft, device, frames, full, load_rows, load_teacher, style

from .common import SEED0, heldout, labelled, load_decoder, load_text, rounded, save_json, synth, teacher_forced
from .stats import paired

WHISPER = "openai/whisper-large-v3-turbo"
SPEAKER = "microsoft/wavlm-base-plus-sv"
LANGUAGE = {"en": "english", "it": "italian"}
CENTROID_ROWS = 40
CONDITIONS = ("teacher_forced", "end_to_end")
TEACHER = "teacher"


def to16k(w):
    from scipy.signal import resample_poly

    return resample_poly(w, 2, 3).astype("float32")


def norm(text, lang):
    t = text.lower().replace("&", " and " if lang == "english" else " e ")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9' ]+", " ", t).replace("'", " ")
    return t.split()


def edits(ref, hyp):
    """Word-level Levenshtein distance (substitutions + deletions + insertions)."""
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1]


@torch.no_grad()
def teacher_training_audio(voice, dev):
    """The teacher's waveform for the first CENTROID_ROWS training rows (int16 round trip, as stored)."""
    _, train = load_rows(voice, train_rows=CENTROID_ROWS)
    teacher, pack = load_teacher(voice, dev)
    cpu_source_stft(teacher.decoder)
    out = []
    for r in train:
        asr, F0, N = frames(r)
        torch.manual_seed(r["seed"])
        y = teacher.decoder(asr[None].to(dev), F0[None].to(dev), N[None].to(dev), style(pack, r["ps"])[:, :128].to(dev))
        out.append(((y.flatten().cpu().clamp(-1, 1) * 32767).short().float() / 32767).numpy())
    del teacher
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("voice", choices=sorted(VOICES))
    ap.add_argument("models", nargs="+", metavar="label=TEXT.pt,DECODER.pt")
    ap.add_argument("--n", type=int, default=200, help="first N held-out rows")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="JSON file (default out/eval/wer-similarity-VOICE-N.json)")
    a = ap.parse_args(argv)
    dev = device(a.device)
    cuda = str(dev).startswith("cuda")
    lang = LANGUAGE[VOICES[a.voice]]
    models = [(label, *paths.split(",")) for label, paths in labelled(a.models)]
    if any(len(m) != 3 for m in models):
        raise SystemExit("expected label=TEXT.pt,DECODER.pt")
    if TEACHER in [m[0] for m in models]:
        raise SystemExit(f"the label {TEACHER} is taken by the teacher's audio")
    rows = heldout(a.voice, a.n)

    # (label, condition) -> waveforms; the teacher's own audio has the condition ""
    audio = {(TEACHER, ""): [full(r)[3].numpy() for r in rows]}
    for label, text_path, dec_path in models:
        ts, dec = load_text(text_path, dev), load_decoder(dec_path, dev)
        audio[label, "teacher_forced"] = [teacher_forced(dec, r, SEED0 + i, dev).cpu().numpy() for i, r in enumerate(rows)]
        audio[label, "end_to_end"] = [synth(ts, dec, r["ids"], SEED0 + i, dev)[0].cpu().numpy() for i, r in enumerate(rows)]
        del ts, dec
    train_audio = teacher_training_audio(a.voice, dev)
    if cuda:
        torch.cuda.empty_cache()
    print(f"{len(rows)} rows rendered, {len(train_audio)} training rows of the teacher", flush=True)

    from transformers import AutoFeatureExtractor, WavLMForXVector, pipeline

    asr = pipeline("automatic-speech-recognition", model=WHISPER, torch_dtype=torch.float16 if cuda else torch.float32, device=dev)
    refs = [norm(r["text"], lang) for r in rows]
    n_words = sum(len(r) for r in refs)
    wer, wer_row, hyp = {}, {}, {}
    for key, waves in audio.items():
        # held-out sentences are shorter than Whisper's 30 s window; longer audio (a broken text side) is chunked
        long = {"chunk_length_s": 30} if max(len(w) for w in waves) > 30 * 24000 else {}
        out = asr([{"raw": to16k(w), "sampling_rate": 16000} for w in waves], batch_size=16,
                  generate_kwargs={"language": lang, "task": "transcribe"}, **long)
        e = [edits(r, norm(h["text"], lang)) for r, h in zip(refs, out)]
        wer[key] = sum(e) / n_words
        wer_row[key] = [x / max(len(r), 1) for x, r in zip(e, refs)]
        hyp[key] = [h["text"].strip() for h in out]
        print(*key, "WER", round(wer[key], 4), flush=True)
    del asr
    if cuda:
        torch.cuda.empty_cache()

    fe = AutoFeatureExtractor.from_pretrained(SPEAKER)
    sv = WavLMForXVector.from_pretrained(SPEAKER).to(dev).eval()

    @torch.no_grad()
    def embed(w):
        inp = fe(to16k(w), sampling_rate=16000, return_tensors="pt")
        e = sv(**{k: v.to(dev) for k, v in inp.items()}).embeddings
        return torch.nn.functional.normalize(e.float(), dim=-1)[0].cpu().numpy()

    centroid = np.mean([embed(w) for w in train_audio], 0)
    centroid /= np.linalg.norm(centroid)
    emb = {key: [embed(w) for w in waves] for key, waves in audio.items()}

    def entry(key):
        to_c = [float(e @ centroid) for e in emb[key]]
        res = {"wer": round(wer[key], 4), "sim_teacher_centroid": round(float(np.mean(to_c)), 4)}
        per_row = {"wer": rounded(wer_row[key]), "sim_teacher_centroid": rounded(to_c)}
        if key[0] != TEACHER:
            same = [float(e @ t) for e, t in zip(emb[key], emb[TEACHER, ""])]
            d = paired(wer_row[key], wer_row[TEACHER, ""])
            res.update(sim_same_sentence_teacher=round(float(np.mean(same)), 4),
                       sim_same_sentence_teacher_se=round(float(np.std(same, ddof=1) / np.sqrt(len(same))), 4),
                       wer_minus_teacher=[d["mean_diff"], d["se"]])
            per_row["sim_same_sentence_teacher"] = rounded(same)
        return {**res, "per_row": per_row, "transcripts": hyp[key]}

    report = {"voice": a.voice, "n": len(rows), "language": lang, "whisper": WHISPER, "speaker_model": SPEAKER,
              "centroid_rows": len(train_audio), TEACHER: entry((TEACHER, ""))}
    for label, text_path, dec_path in models:
        report[label] = {"text": text_path, "decoder": dec_path, **{c: entry((label, c)) for c in CONDITIONS}}
    for label, cond in audio:
        e = report[label][cond] if cond else report[label]
        print(label, cond, {k: v for k, v in e.items() if k not in ("per_row", "transcripts")}, flush=True)
    save_json(a.out or f"out/eval/wer-similarity-{a.voice}-{len(rows)}.json", report)


if __name__ == "__main__":
    main()
