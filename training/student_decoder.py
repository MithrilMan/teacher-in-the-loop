# Derived from Paradee's training/scripts/student_decoder.py (Apache-2.0); see NOTICE.
"""Acoustic-side student: Kokoro's Decoder with the hard-coded 1024/512 widths made configurable, generator sized
via the same istftnet config keys. Preset A is Paradee's decoder (3.85M parameters), used unchanged in the paper."""
import torch, torch.nn as nn
from torch.nn.utils import weight_norm
from kokoro.istftnet import Decoder, AdainResBlk1d, Generator

DEC_PRESETS = {  # name: (adain width, upsample_initial_channel, resblock kernels)
    "teacher": (1024, 512, [3, 7, 11]),
    "A": (256, 128, [3, 7, 11]),
}


class StudentDecoder(Decoder):
    def __init__(self, preset="A", dim_in=512, style_dim=16, dim_out=80, disable_complex=False):
        W, uic, kernels = DEC_PRESETS[preset]
        nn.Module.__init__(self)
        self.encode = AdainResBlk1d(dim_in + 2, W, style_dim)
        self.decode = nn.ModuleList([AdainResBlk1d(W + 2 + 64, W, style_dim) for _ in range(3)]
                                    + [AdainResBlk1d(W + 2 + 64, uic, style_dim, upsample=True)])
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1))
        self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(dim_in, 64, kernel_size=1)))
        self.generator = Generator(style_dim, kernels, [10, 6], uic, [[1, 3, 5]] * len(kernels), [20, 12], 20, 5,
                                   disable_complex=disable_complex)
        self.style = nn.Parameter(torch.randn(1, style_dim) * 0.1)  # the one voice

    def forward(self, asr, F0_curve, N, s=None):
        return super().forward(asr, F0_curve, N, self.style.expand(asr.shape[0], -1))


def load_student_decoder(path, device="cpu"):
    """A trained decoder. Its source module holds the teacher's weights, copied in before training."""
    ck = torch.load(path, map_location=device, weights_only=False)
    m = StudentDecoder("A").to(device).eval()
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    return m
