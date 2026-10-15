"""Runs a real graph when TITL_TEST_MODEL points to one (and TITL_TEST_LANGUAGE gives its language, default "it")."""
import os
import wave

import numpy as np
import pytest

from titl import SAMPLE_RATE, Voice, write_wav

MODEL = os.environ.get("TITL_TEST_MODEL")
LANGUAGE = os.environ.get("TITL_TEST_LANGUAGE", "it")
TEXT = {"en": "The quick brown fox jumps over the lazy dog. It was a sunny day.",
        "it": "Il gatto dorme sul divano. Oggi è una bella giornata di sole."}

needs_model = pytest.mark.skipif(not MODEL, reason="set TITL_TEST_MODEL to an exported graph")


@needs_model
def test_synthesis_and_speed():
    voice = Voice(MODEL, LANGUAGE)
    audio = voice(TEXT[LANGUAGE])
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert 2.0 < len(audio) / SAMPLE_RATE < 10.0
    assert np.isfinite(audio).all() and 0.01 < np.abs(audio).max() <= 1.5
    faster = voice(TEXT[LANGUAGE], speed=1.25)
    assert len(faster) < len(audio)


@needs_model
def test_long_text_is_chunked():
    voice = Voice(MODEL, LANGUAGE)
    sentence = TEXT[LANGUAGE].split(".")[0] + ", "
    audio = voice(sentence * 40 + "fine.")  # one sentence of more than 510 phonemes
    assert len(audio) / SAMPLE_RATE > 30


def test_write_wav(tmp_path):
    path = tmp_path / "x.wav"
    write_wav(path, np.sin(np.linspace(0, 100, SAMPLE_RATE)).astype(np.float32) * 2)
    with wave.open(str(path)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (SAMPLE_RATE, 1, 2, SAMPLE_RATE)


@needs_model
def test_empty_text_gives_no_audio():
    assert len(Voice(MODEL, LANGUAGE)("  \n ")) == 0
