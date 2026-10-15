# Derived from Paradee's training/scripts/phase_lock.py (Apache-2.0); see NOTICE.
"""Paradee's phase-lock post-filter, which its released graph contains and ours do not: in voiced frames, the
phase of every STFT bin between 2 and 8 kHz becomes the harmonic source's phase plus a time-smoothed version of
(output phase - source phase); magnitudes are kept. Used only to evaluate it (paper, "phase-locking post-filter")."""
import torch
import torch.nn.functional as F

n, hop, SR = 1024, 256, 24000
W = torch.hann_window(n)
FR = torch.arange(n // 2 + 1) * SR / n
BAND = ((FR >= 2000) & (FR < 8000))[:, None]


def smooth(Z, L):
    k = torch.hann_window(L + 2)[1:-1]
    k = (k / k.sum())[None, None]
    return torch.complex(F.conv1d(Z.real[:, None], k, padding=L // 2)[:, 0], F.conv1d(Z.imag[:, None], k, padding=L // 2)[:, 0])


def lock(s, src, f0, L=33):
    """s: a waveform on the CPU, src: the harmonic excitation it was rendered with, f0: its F0 (80 values per second).
    L = 33 frames is the smoothing of Paradee's exported graph."""
    k = min(len(s), len(src))
    s, src = s[:k], src[:k]
    S = torch.stft(s, n, hop, window=W, return_complex=True)
    E = torch.stft(src, n, hop, window=W, return_complex=True)
    v = torch.tensor([float(f0[min(j * hop // 300, len(f0) - 1)]) > 60 for j in range(S.shape[1])])[None]
    rel = S * E.conj() / E.abs().clamp_min(1e-6)
    ph = torch.where(BAND & v, E.angle() + smooth(rel, L).angle(), S.angle())
    return torch.istft(torch.polar(S.abs(), ph), n, hop, window=W, length=k)
