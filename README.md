<h1 align="center">Teacher in the Loop</h1>

<p align="center"><b>Reliable Distillation of Kokoro-82M into 8M-Parameter Single-Voice Students, in English and Italian</b></p>

<p align="center">
  <a href="paper/Teacher-in-the-Loop-Angela-2026.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg" alt="Paper"></a>
  <a href="https://doi.org/10.5281/zenodo.23400040"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23400040-1682D4.svg" alt="DOI"></a>
  <a href="https://huggingface.co/spaces/MithrilMan/teacher-in-the-loop"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo-runs%20in%20your%20browser-blue" alt="Demo"></a>
  <a href="https://huggingface.co/spaces/MithrilMan/teacher-in-the-loop#listen"><img src="https://img.shields.io/badge/%F0%9F%94%8A%20Audio-samples-blueviolet" alt="Audio samples"></a>
  <a href="https://huggingface.co/MithrilMan/teacher-in-the-loop"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-int8%20ONNX-yellow" alt="Weights"></a>
  <a href="https://pypi.org/project/teacher-in-the-loop/"><img src="https://img.shields.io/badge/PyPI-teacher--in--the--loop-3775A9.svg?logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green.svg" alt="License"></a>
</p>

<p align="center"><em>Kokoro-82M's voices <code>af_heart</code>, <code>if_sara</code> and <code>im_nicola</code> as 8M-parameter
students: one 9 MB int8 ONNX graph per voice, more than twenty times faster than real time on one CPU thread.</em></p>

---

Paper, training code and inference code for **"Teacher in the Loop: Reliable Distillation of Kokoro-82M into 8M-Parameter
Single-Voice Students, in English and Italian"** (Fabio Angela, 2026), a follow-up to Paradee.

- **[Try it in your browser](https://huggingface.co/spaces/MithrilMan/teacher-in-the-loop)**: the three voices run
  on your device (ONNX Runtime Web and eSpeak NG compiled to WebAssembly); nothing you type leaves the browser.
- **[Listen to the paper's samples](https://huggingface.co/spaces/MithrilMan/teacher-in-the-loop#listen)**: the teacher,
  Paradee's released decoder, the published recipe and ours on the same held-out sentences.
- **[Download the weights](https://huggingface.co/MithrilMan/teacher-in-the-loop)**: int8 and float32 graphs and the
  PyTorch weights of each voice.

The paper is on Zenodo ([10.5281/zenodo.23400040](https://doi.org/10.5281/zenodo.23400040)); an arXiv version
will follow. The training package was checked against the paper's numbers on a GPU run (paper, Appendix G).

## Results at a glance

UTMOS of the decoder alone on the teacher's own features, mean over 200 held-out sentences:

| Voice | Kokoro-82M (teacher) | Paradee, released | Published recipe, our run | **This paper** |
|---|---|---|---|---|
| `af_heart` (English) | 4.52 | 4.37 | 2.89 | **4.47** |
| `if_sara` (Italian) | 4.02 | – | 2.21 | **3.88** |
| `im_nicola` (Italian) | 3.87 | – | 3.53 | **3.70** |

End to end, with the existing text students, the scores do not drop, and int8 export with per-channel scales
costs at most 0.01 UTMOS. Paradee released only an English voice.

## Speak

```bash
pip install teacher-in-the-loop
titl --voice if_sara "Buongiorno a tutti." -o buongiorno.wav
```

```python
from titl import Voice, write_wav

voice = Voice.from_pretrained("af_heart")      # or "if_sara", "im_nicola"; int8 by default
write_wav("hello.wav", voice("Hello there. This voice has eight million parameters.", speed=1.0))
```

Each voice is one ONNX graph of about 9 MB (int8 weights) that runs more than twenty times faster than real time
on one CPU thread. Text becomes phonemes as in Kokoro: Misaki for English, Misaki's eSpeak NG backend for Italian.
English also needs spaCy's `en_core_web_sm` model, which Misaki downloads with pip on first use; to work offline
or without pip, install it beforehand (`python -m spacy download en_core_web_sm`).

## Layout

```
src/titl/     inference: phonemes, the ONNX voice, command line (pip package teacher-in-the-loop)
training/     the paper's recipe: teacher corpus, text side, teacher pool, decoder stages, export
eval/         evaluation behind the paper's tables, held-out sentences, speed benchmark
tests/        tests of the inference package
paper/        the paper (PDF) and data/, the CSVs behind every table and figure with their sources
hf/           model card of the Hugging Face weights repository
space/        the Hugging Face Space: in-browser demo (JavaScript phonemes, ONNX Runtime Web) and audio samples
LICENSE       Apache-2.0 (Paradee, from which the training code derives, is Apache-2.0), except the sentence
              lists in training/data/ and eval/heldout.json: CC BY-SA 4.0 (training/data/LICENSE.md)
NOTICE        attribution for Paradee and Kokoro-82M
THIRD_PARTY_NOTICES.md
              third-party models, code, Python packages and data, with their licenses
CITATION.cff  how to cite this repository and the paper
```

Converting text to phonemes loads eSpeak NG and phonemizer-fork (GPL-3.0-or-later) and num2words (LGPL-2.1) at
run time. pip installs them as separate packages; this repository does not include them
(`THIRD_PARTY_NOTICES.md`, section 3).

## Scope

Only the distillation of Kokoro-82M's voices `af_heart` (English), `if_sara` and `im_nicola` (Italian): the
revised decoder recipe described in the paper, the existing Paradee text side, the int8 ONNX export, the
evaluation needed to reproduce the paper's tables, and the inference code that runs the released voices.

## The paper

`paper/Teacher-in-the-Loop-Angela-2026.pdf`, also on Zenodo as
[10.5281/zenodo.23400040](https://doi.org/10.5281/zenodo.23400040); the LaTeX source will be on arXiv. Every number
in it is in `paper/data/`, which the paper calls `data/`; `paper/data/SOURCES.md` says where each one comes from.

## Citation

```bibtex
@misc{angela2026teacher,
  author    = {Fabio Angela},
  title     = {Teacher in the Loop: Reliable Distillation of {Kokoro-82M} into {8M}-Parameter Single-Voice
               Students, in {English} and {Italian}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23400040},
  url       = {https://doi.org/10.5281/zenodo.23400040}
}
```
