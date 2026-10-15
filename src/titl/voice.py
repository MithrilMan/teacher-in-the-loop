# Follows Paradee's paradee/tts.py (Apache-2.0); see NOTICE.
"""Phonemes -> token ids -> one ONNX graph -> 24 kHz audio."""
from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np

from .phonemes import Phonemizer, encode, load_vocab

SAMPLE_RATE = 24000
REPO = "MithrilMan/teacher-in-the-loop"
REVISION = "cca42f3882e9d36b865883298d8e3c6a6e42b9c0"  # commit of the weights repository with the paper's graphs

# Kokoro voice -> language of the student trained from it
VOICES = {"af_heart": "en", "if_sara": "it", "im_nicola": "it"}
INT8_FILE, FP32_FILE = "model_int8.onnx", "model.onnx"


class Voice:
    """One distilled voice: call it with text to get a float32 waveform at 24 kHz.

    model_path: an exported graph (inputs ``input_ids`` int64 [1, T] and ``speed`` float32 [1], output
    ``waveform`` float32 [1, N]). language: "en" or "it". threads: CPU threads for ONNX Runtime; one
    thread already runs more than twenty times faster than real time.
    """

    def __init__(self, model_path: str | os.PathLike, language: str, *, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        inputs = {i.name for i in self.session.get_inputs()}
        outputs = [o.name for o in self.session.get_outputs()]
        if inputs != {"input_ids", "speed"} or outputs[:1] != ["waveform"]:
            raise ValueError(f"{model_path}: expected inputs input_ids, speed and output waveform, "
                             f"found {sorted(inputs)} -> {outputs}")
        self.language = language
        self.phonemizer = Phonemizer(language)
        self.vocab = load_vocab()

    @classmethod
    def from_pretrained(cls, name: str, *, quantized: bool = True, revision: str = REVISION,
                        threads: int = 1) -> "Voice":
        """Download one of the released voices (``af_heart``, ``if_sara``, ``im_nicola``) and load it.

        quantized: the int8 graph (default, about 9 MB) instead of the float32 one (about 33 MB).
        """
        if name not in VOICES:
            raise ValueError(f"unknown voice {name!r}; available: {', '.join(VOICES)}")
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(REPO, f"{name}/{INT8_FILE if quantized else FP32_FILE}", revision=revision)
        return cls(path, VOICES[name], threads=threads)

    def from_phonemes(self, phonemes: str, speed: float = 1.0) -> np.ndarray:
        """Audio for one chunk of at most 510 phonemes."""
        feed = {"input_ids": np.array([encode(phonemes, self.vocab)], dtype=np.int64),
                "speed": np.array([speed], dtype=np.float32)}
        return self.session.run(["waveform"], feed)[0][0]

    def __call__(self, text: str, speed: float = 1.0) -> np.ndarray:
        """speed > 1 speaks faster. Long text is read one sentence at a time."""
        parts = [self.from_phonemes(ps, speed) for ps in self.phonemizer.chunks(text)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def write_wav(path: str | os.PathLike, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Save mono float audio in [-1, 1] as 16-bit PCM."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(Path(path)), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
