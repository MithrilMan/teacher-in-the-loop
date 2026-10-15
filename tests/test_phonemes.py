import json
from pathlib import Path

import pytest

from titl.phonemes import MAX_PHONEMES, Phonemizer, encode, load_vocab, split_phonemes


def test_vocab_is_kokoro():
    vocab = load_vocab()
    assert len(vocab) == 114
    assert vocab[" "] == 16 and vocab["t"] == 62
    assert "̪" not in vocab


def test_encode_pads_and_drops_unknown():
    vocab = load_vocab()
    assert encode("t a", vocab) == [0, vocab["t"], vocab[" "], vocab["a"], 0]
    assert encode("t̪", vocab) == [0, vocab["t"], 0]
    with pytest.raises(ValueError):
        encode("a" * (MAX_PHONEMES + 1), vocab)


def test_split_at_last_space():
    words = " ".join(["abcd"] * 300)  # 1499 symbols
    parts = split_phonemes(words, 100)
    assert all(len(p) <= 100 for p in parts)
    assert " ".join(parts) == words
    assert split_phonemes("x" * 250, 100) == ["x" * 100, "x" * 100, "x" * 50]


def test_italian_dental_stops():
    ph = Phonemizer("it")
    out = ph("La FIAT produce automobili. Il software è nuovo.")
    assert "̪" not in out
    assert set(out) <= set(load_vocab())


def test_italian_sentences():
    chunks = Phonemizer("it").chunks("Buongiorno a tutti. Come state?\nBene, grazie!")
    assert len(chunks) == 3


def test_english():
    out = Phonemizer("en")("Hello there, how are you?")
    assert out and set(out) <= set(load_vocab())


def test_same_ids_as_the_teacher_data():
    """The 200 held-out sentences per language: the ids this package produces are the ones the teacher data used."""
    rows = json.loads((Path(__file__).parents[1] / "eval" / "heldout.json").read_text(encoding="utf-8"))
    vocab = load_vocab()
    for language, items in rows.items():
        ph = Phonemizer(language)
        bad = [r["text"] for r in items if encode(ph(r["text"]), vocab) != r["ids"]]
        assert not bad, f"{language}: {len(bad)} of {len(items)} differ, e.g. {bad[0]!r}"
