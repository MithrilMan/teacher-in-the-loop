"""UTMOS (utmos22_strong, SpeechMOS implementation, pinned commit), the paper's main metric."""
import torch

SPEECHMOS = "tarepan/SpeechMOS:ed25eacbfa42b99156c36ebec67a733b5dbb9b79"


class Utmos:
    def __init__(self, dev):
        self.dev = dev
        self.m = torch.hub.load(SPEECHMOS, "utmos22_strong", trust_repo=True, progress=False).to(dev).eval()

    @torch.no_grad()
    def __call__(self, wav):
        """wav: a 24 kHz waveform [samples]."""
        return self.m(wav.float()[None].to(self.dev), 24000).item()
