"""Text to Kokoro phonemes and token ids, as Kokoro's own pipeline does it.

English uses Misaki's G2P with eSpeak NG as fallback (``kokoro.KPipeline(lang_code="a")``). Italian uses
Misaki's eSpeak NG backend (``lang_code="i"``). eSpeak marks Italian dental stops with U+032A COMBINING
BRIDGE BELOW, which Kokoro's vocabulary lacks; they are mapped to plain ``t`` and ``d``. The teacher corpus,
the students and this package all use the same conversion.
"""
from __future__ import annotations

import json
import re
import threading
from importlib import resources

LANGUAGES = ("en", "it")
MAX_PHONEMES = 510  # the text side has 512 positions, two of which hold the padding token 0

_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+|\n+")
_DENTAL = str.maketrans({"̪": None})


def load_vocab() -> dict[str, int]:
    """Kokoro-82M's phoneme vocabulary (from its ``config.json``), shared by every voice."""
    with resources.files("titl").joinpath("vocab.json").open(encoding="utf-8") as f:
        return json.load(f)


class Phonemizer:
    """Convert text to phonemes for one language. eSpeak is not thread-safe, so calls are serialized."""

    def __init__(self, language: str):
        if language not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}, not {language!r}")
        self.language = language
        self._lock = threading.Lock()
        if language == "en":
            from misaki import en, espeak

            self._g2p = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False), unk="")
        else:
            from misaki import espeak

            self._g2p = espeak.EspeakG2P(language="it")

    def __call__(self, text: str) -> str:
        with self._lock:
            phonemes, _ = self._g2p(text)
        if self.language == "it":
            phonemes = phonemes.translate(_DENTAL)  # t̪ d̪ -> t d
        return phonemes.strip()

    def chunks(self, text: str, max_phonemes: int = MAX_PHONEMES) -> list[str]:
        """Phonemes of ``text`` one sentence at a time; a sentence that is too long is cut at a space."""
        out = []
        for sentence in _SENTENCE_BREAK.split(text.strip()):
            if not sentence.strip():
                continue
            out.extend(split_phonemes(self(sentence), max_phonemes))
        return out


def split_phonemes(phonemes: str, max_phonemes: int = MAX_PHONEMES) -> list[str]:
    """Cut a phoneme string into pieces of at most ``max_phonemes``, at the last space that fits."""
    out = []
    rest = phonemes.strip()
    while len(rest) > max_phonemes:
        cut = rest.rfind(" ", 0, max_phonemes + 1)
        cut = cut if cut > 0 else max_phonemes
        out.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        out.append(rest)
    return out


def encode(phonemes: str, vocab: dict[str, int]) -> list[int]:
    """Token ids with Kokoro's padding token at both ends. Symbols outside the vocabulary are dropped, as in Kokoro."""
    ids = [vocab[c] for c in phonemes if c in vocab]
    if len(ids) > MAX_PHONEMES:
        raise ValueError(f"{len(ids)} phonemes; at most {MAX_PHONEMES} fit in one call")
    return [0, *ids, 0]
