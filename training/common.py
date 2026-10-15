"""Shared pieces: voices, the Kokoro teacher, teacher rows, crops, the spectral objective and checkpoints."""
import copy
import glob
import json
import os
import random

import torch
import torch.nn.functional as F
import torchaudio

TEACHER_REPO = "hexgrad/Kokoro-82M"
TEACHER_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
VOICES = {"af_heart": "en", "if_sara": "it", "im_nicola": "it"}
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SENTENCES = {"en": os.path.join(DATA, "wikitext_sents.json"), "it": os.path.join(DATA, "italian_sents.json")}
SR = 24000
HOP = 600  # waveform samples per decoder frame (40 frames per second; F0 and N run at twice that rate)


def device(name=None):
    name = name or ("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return name


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- teacher
def teacher_files(voice):
    from huggingface_hub import hf_hub_download

    get = lambda name: hf_hub_download(TEACHER_REPO, name, revision=TEACHER_REVISION)
    return get("config.json"), get("kokoro-v1_0.pth"), get(f"voices/{voice}.pt")


def load_teacher(voice, dev="cpu"):
    """Kokoro-82M (pinned revision) with weight norm kept, and the voice pack [510, 1, 256]."""
    from kokoro import KModel

    config, weights, pack = teacher_files(voice)
    model = KModel(repo_id=TEACHER_REPO, config=config, model=weights).to(dev).eval()
    return model, torch.load(pack, map_location="cpu", weights_only=True)


def style(pack, ps):
    """Kokoro indexes the voice pack by phoneme count; the decoder takes the first 128 dimensions."""
    return pack[len(ps) - 1]


def cpu_source_stft(decoder):
    """Compute the harmonic source's STFT on the CPU. Its phase feature sits on the +-pi branch cut for strong
    harmonic bins, and its sign there differs between devices; on the CPU the result is canonical."""
    st = decoder.generator.stft
    orig = type(st).transform

    def transform(x):
        mag, ph = orig(st, x.cpu())
        return mag.to(x.device), ph.to(x.device)

    st.transform = transform
    return decoder


def gpu_source_stft(decoder):
    """The same STFT on the device, without the CPU round trip. DC and Nyquist bins of a real signal are real:
    their imaginary part is set to +0.0, so their phase is exactly 0 or +pi, as on the CPU."""
    st = decoder.generator.stft

    def transform(x):
        X = torch.stft(x, st.filter_length, st.hop_length, st.win_length, window=st.window.to(x.device), return_complex=True)
        re, im = X.real, X.imag.clone()
        im[:, 0], im[:, -1] = 0.0, 0.0
        return X.abs(), torch.atan2(im, re)

    st.transform = transform
    return decoder


class FixedSource(torch.nn.Module):
    """Stands in for the generator's source module: returns an excitation set from outside."""

    def __init__(self):
        super().__init__()
        self.src = None

    def forward(self, f0):
        return self.src, None, None


class Excitation:
    """The teacher's own source module (sine harmonics plus noise, mixed by a learned layer) on the device."""

    def __init__(self, teacher, dev):
        g = teacher.decoder.generator
        self.m_source = copy.deepcopy(g.m_source).to(dev)
        self.f0_upsamp = copy.deepcopy(g.f0_upsamp).to(dev)

    @torch.no_grad()
    def __call__(self, F0):
        """F0 [B, 2F] -> excitation [B, 600F, 1]."""
        return self.m_source(self.f0_upsamp(F0[:, None]).transpose(1, 2))[0]


# ---------------------------------------------------------------- teacher rows
def run_dir(voice):
    return os.path.join("runs", voice)


def load_rows(voice, train_rows=0, keep_audio=False):
    """Held-out rows and training rows of a voice's teacher corpus (gen_teacher.py). train_rows > 0 keeps only
    the first train_rows training rows. Training rows drop their audio unless keep_audio."""
    d = os.path.join(run_dir(voice), "teacher")
    held = torch.load(os.path.join(d, "heldout.pt"), weights_only=False)
    train = []
    for path in sorted(glob.glob(os.path.join(d, "shard_*.pt"))):
        rows = torch.load(path, weights_only=False)
        for r in rows:
            r.pop("d", None)
            if not keep_audio:
                r.pop("audio", None)
        train += rows
        if 0 < train_rows <= len(train):
            break
    if not train:
        raise FileNotFoundError(f"no teacher rows in {d}; run gen_teacher.py {voice} first")
    return held, train[:train_rows] if train_rows > 0 else train


def frames(r):
    """Decoder inputs of a whole sentence: phoneme features expanded to frames [512, F], F0 and N [2F]."""
    idx = torch.repeat_interleave(torch.arange(len(r["pred_dur"])), r["pred_dur"])
    return r["t_en"].float()[:, idx], r["F0"][: 2 * len(idx)], r["N"][: 2 * len(idx)]


def full(r):
    """frames(r) plus the teacher's waveform for the sentence."""
    asr, F0, N = frames(r)
    return asr, F0, N, r["audio"][: HOP * asr.shape[1]].float() / 32767


def crop(r, seg, start=None):
    """A crop of seg frames from a random (or the given) start; a shorter sentence is zero-padded at the end."""
    asr, F0, N = frames(r)
    Fr = asr.shape[1]
    if Fr <= seg:
        pad = seg - Fr
        return F.pad(asr, (0, pad)), F.pad(F0, (0, 2 * pad)), F.pad(N, (0, 2 * pad)), 0
    a = random.randint(0, Fr - seg) if start is None else start
    return asr[:, a : a + seg], F0[2 * a : 2 * (a + seg)], N[2 * a : 2 * (a + seg)], a


# ---------------------------------------------------------------- objective (Paradee's, unchanged)
_MEL = {}


def spec_losses(y, t):
    """Log-mel L1 (FFT 1024, hop 256, 80 bands) and the multi-resolution STFT term (FFT 512/1024/2048, hop FFT/4:
    spectral convergence plus log-magnitude L1)."""
    if y.device not in _MEL:
        _MEL[y.device] = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=256, n_mels=80).to(y.device)
    mel = _MEL[y.device]
    lm = F.l1_loss(torch.log(mel(y).clamp_min(1e-5)), torch.log(mel(t).clamp_min(1e-5)))
    ls = 0
    for n in (512, 1024, 2048):
        w = torch.hann_window(n, device=y.device)
        Y = torch.stft(y, n, n // 4, window=w, return_complex=True).abs()
        T = torch.stft(t, n, n // 4, window=w, return_complex=True).abs()
        ls = ls + (T - Y).norm() / T.norm().clamp_min(1e-5) + F.l1_loss(torch.log(Y.clamp_min(1e-5)), torch.log(T.clamp_min(1e-5)))
    return lm, ls / 3


# ---------------------------------------------------------------- checkpoints
def save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def log_json(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
