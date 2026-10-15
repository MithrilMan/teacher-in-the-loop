---
license: apache-2.0
language:
- en
- it
tags:
- text-to-speech
- tts
- speech-synthesis
- onnx
- int8
- on-device
- kokoro
- distillation
- knowledge-distillation
pipeline_tag: text-to-speech
---

# Teacher in the Loop: 8M-parameter students of Kokoro-82M

**[Try it in your browser](https://huggingface.co/spaces/MithrilMan/teacher-in-the-loop)** ·
[Audio samples](https://huggingface.co/spaces/MithrilMan/teacher-in-the-loop#listen) ·
[Code](https://github.com/MithrilMan/teacher-in-the-loop) ·
[Paper](https://github.com/MithrilMan/teacher-in-the-loop/blob/main/paper/Teacher-in-the-Loop-Angela-2026.pdf) ·
[PyPI](https://pypi.org/project/teacher-in-the-loop/)

Three single-voice text-to-speech models of 8.07M parameters, distilled from
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) with the revised recipe of the paper "Teacher in the
Loop: Reliable Distillation of Kokoro-82M into 8M-Parameter Single-Voice Students, in English and Italian"
(Fabio Angela, 2026). Each model speaks one of Kokoro's voices: `af_heart` (English), `if_sara` and
`im_nicola` (Italian).

The architecture is Paradee's ([code](https://github.com/sahilmahendrakar/paradee),
[model](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.0)), unchanged: a text side of 4.23M parameters that predicts
durations, pitch, energy and phoneme features, and a decoder of 3.85M parameters that turns them into a 24 kHz
waveform. What changed is how the decoder is trained: the frozen teacher renders each training crop with the
excitation the student receives ("teacher in the loop"), the decoder also learns the teacher's inverse-STFT
head, and the final adversarial stage runs for 30,000 steps (the paper's "final recipe"). Each voice is one ONNX
graph from phoneme ids to audio, 9.17 MB with int8 weights, running 24 to 26 times faster than real time on
one CPU thread.

Code, training recipe and paper: https://github.com/MithrilMan/teacher-in-the-loop

## Voices

UTMOS (`utmos22_strong`) on each voice's 200 held-out sentences. "Decoder alone": the decoder receives the
teacher's own durations, pitch, energy and phoneme features. "End to end": text side and decoder, from phonemes,
in PyTorch. "Int8": the exported graphs, ONNX Runtime on the CPU, end to end on the first 100 held-out
sentences, int8 minus float32 (paired, mean ± standard error).

| voice | language | Kokoro teacher | decoder alone (95% CI) | end to end | int8 − float32 |
|---|---|---|---|---|---|
| `af_heart` | English | 4.521 | 4.467 ± 0.006 | 4.474 | −0.006 ± 0.001 |
| `if_sara` | Italian | 4.022 | 3.882 ± 0.025 | 3.895 | −0.007 ± 0.011 |
| `im_nicola` | Italian | 3.867 | 3.696 ± 0.021 | 3.700 | −0.010 ± 0.005 |

On the same 200 English sentences, Paradee's released decoder scores 4.371 (decoder alone). Read the Italian
numbers with the caveats below: UTMOS was trained on English ratings.

## Usage

With the `titl` package (Python 3.10 to 3.12):

```bash
pip install teacher-in-the-loop
titl --voice if_sara "Buongiorno a tutti." -o buongiorno.wav
titl --voice af_heart --speed 1.1 "Hello there." -o hello.wav   # --fp32 for the float32 graph
```

```python
from titl import Voice, write_wav

voice = Voice.from_pretrained("af_heart")       # or "if_sara", "im_nicola"; int8 graph by default
audio = voice("Hello there. This voice has eight million parameters.", speed=1.0)   # float32, 24 kHz
write_wav("hello.wav", audio)

voice32 = Voice.from_pretrained("im_nicola", quantized=False)   # float32 graph
```

`Voice.from_pretrained` downloads `<voice>/model_int8.onnx` (or `model.onnx`) from this repository. Text is
converted to phonemes as Kokoro does: Misaki 0.9.4 for English, Misaki with eSpeak NG for Italian, with the
dental stops t̪ and d̪ mapped to plain t and d. Long text is read one sentence at a time, in chunks of at most
510 phonemes. English also needs spaCy's `en_core_web_sm` model, which Misaki downloads with pip on first use; to
work offline or without pip, install it beforehand (`python -m spacy download en_core_web_sm`).

The graphs can also be used without the package, from phonemes:

```python
import json
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

repo = "MithrilMan/teacher-in-the-loop"
vocab = json.load(open(hf_hub_download(repo, "vocab.json"), encoding="utf-8"))
session = ort.InferenceSession(hf_hub_download(repo, "if_sara/model_int8.onnx"), providers=["CPUExecutionProvider"])

from titl.phonemes import Phonemizer           # or any source of Kokoro phonemes
phonemes = Phonemizer("it")("Buongiorno a tutti.")
ids = [0] + [vocab[c] for c in phonemes if c in vocab] + [0]
waveform = session.run(["waveform"], {"input_ids": np.array([ids], dtype=np.int64),
                                      "speed": np.array([1.0], dtype=np.float32)})[0][0]
```

## ONNX contract

| | name | type | shape | content |
|---|---|---|---|---|
| input | `input_ids` | int64 | `[1, T]` | phoneme ids from `vocab.json` with the padding id 0 at both ends; at most 510 phonemes (T ≤ 512) |
| input | `speed` | float32 | `[1]` | 1.0 is the teacher's pace; above 1 speaks faster |
| output | `waveform` | float32 | `[1, N]` | mono audio at 24 kHz |

The excitation is random, so two runs on the same input differ slightly. The graphs contain no post-filter
(see below).

## Files

| file | size | content |
|---|---|---|
| `vocab.json` | 1 KB | Kokoro-82M's phoneme vocabulary (from its `config.json`), shared by all voices |
| `<voice>/model_int8.onnx` | 9.17 MB | the voice as one graph, int8 weights with one scale per output channel (biases, norms and windows in float32), float32 arithmetic |
| `<voice>/model.onnx` | 32.8 MB | the same graph with float32 weights |
| `<voice>/text.safetensors` | 16.9 MB | PyTorch weights of the text side, `training.student.TextStudent("s+mlp")` |
| `<voice>/decoder.safetensors` | 15.4 MB | PyTorch weights of the decoder, `training.student_decoder.StudentDecoder("A")` |
| `<voice>/config.json` | | voice, language, sample rate, ONNX contract, phoneme settings, teacher revision, headline results |
| `<voice>/export-validation.json` | | the exporter's CPU check of both graphs: output lengths, timing, int8 parity |
| `LICENSE` | | the Apache License 2.0 |
| `SHA256SUMS` | | SHA-256 of every other file |

The PyTorch files hold model weights only, no optimizer state. The ONNX graphs were exported from exactly these
weights. To use them with the training and evaluation code of the repository, which reads checkpoints with a
`model` key:

```python
import torch
from safetensors.torch import load_file
from training.student import TextStudent
from training.student_decoder import StudentDecoder

text = TextStudent("s+mlp")
text.load_state_dict(load_file("if_sara/text.safetensors"))
decoder = StudentDecoder("A")
decoder.load_state_dict(load_file("if_sara/decoder.safetensors"))
torch.save({"model": decoder.state_dict()}, "decoder.pt")   # e.g. for python -m training.export
```

## Training

- **Teacher:** Kokoro-82M v1.0, revision `f3ff3571791e39611d31c381e3a41a3af07b4987`, with the voice pack of the
  student's voice.
- **Data:** synthetic speech only. Kokoro-82M renders sentences from WikiText-103 for `af_heart` (11,999
  sentences, the first 200 held out; the decoder used 5,799 training sentences) and 12,000 training and 200
  held-out sentences from 1,203 articles of the Italian Wikipedia (snapshot `20231101.it` of
  `wikimedia/wikipedia`, no article in both splits) for `if_sara` and `im_nicola`. No recorded human speech is
  used. The decoder trains on a pool of 150,000 random 64-frame crops rendered by the frozen teacher.
- **Decoder, final recipe:** stage one for 30,000 steps from scratch (spectral losses plus distillation of the
  teacher's inverse-STFT head, with the teacher's source module copied in and frozen), a warm restart of 20,000
  steps, then the adversarial stages GAN10 (5,000 steps) and GAN3 (30,000 steps).
- **Text side:** Paradee's `s+mlp` student trained with the upstream text recipe (8,000 steps), not retrained for
  these models.
- **Compute:** one shared 32 GB GPU; about eleven to twelve hours of training per voice, plus about one hour and
  62 GB of disk to render the pool.

`python -m training.recipe VOICE` in the code repository runs the whole recipe. The paper checked the training
package on `if_sara` only (3.878 against 3.882 for the released decoder); the int8 cost varied between trainings
there, so measure it for each new voice.

## Evaluation and limitations

- **One automatic metric.** Almost every number is a UTMOS prediction. UTMOS was trained on English ratings; the
  Italian teachers themselves score only 4.02 and 3.87, and Kokoro's documentation grades both Italian voices C.
  Read the Italian numbers as a comparison between systems on the same voice, not as absolute quality. In
  English UTMOS is close to saturation: the released Paradee decoder, Paradee's paper model and all the paper's
  English decoders after the adversarial stages lie between 4.36 and 4.47.
- **No human MOS.** The listening evidence is two blind four-sentence comparisons in English by a single
  listener, the author, plus informal listening. A faint metallic ring added by the small decoder is the main
  audible defect known.
- **Harmonic definition above 2 kHz.** The students keep the teacher's harmonic contrast below 1 kHz, but
  between 2 and 4 kHz they reach 5.5 dB (`af_heart`) and 7.7 dB (`if_sara`) against the teacher's 10.5 and
  11.8 dB: they fill part of the space between the upper harmonics with noise. UTMOS does not register this.
- **Bounded by the teacher.** The students imitate Kokoro, including its mispronunciations, its prosody and the
  errors of its text-to-phoneme conversion. Whisper large-v3-turbo transcribes them end to end with word error
  rates of 3.29%, 3.94% and 4.33% (teachers 3.32%, 4.09% and 4.45%), and their speaker similarity to the
  teacher's audio of the same sentence is 0.990, 0.995 and 0.993.
- **Seeds and held-out set.** Each voice was trained once, except `if_sara`, whose final recipe was repeated
  with two more seeds (3.882, 3.881 and 3.883). The 200 held-out sentences also served for the paper's design
  decisions, so they are a development set, not an untouched test set.
- **Int8.** The int8 graphs cost at most 0.01 UTMOS end to end, but they fail the exporter's parity gate
  (`af_heart` 2.45 vs 1.31 dB, `if_sara` 1.83 vs 1.37 dB, `im_nicola` 1.46 vs 1.04 dB of median log-spectral
  distance), because int8 rounding moves 1 to 4 of 546 token durations by one frame and the gate only compares
  cases where all durations match. The gate is reported in `export-validation.json`, not enforced. These are the
  graphs evaluated in the paper.
- **No phase-locking filter.** Unlike Paradee's released graph, which contains its 2–8 kHz phase-locking
  post-filter, these graphs have no post-filter: the filter lowers the score of decoders trained with this
  recipe (by 0.03 to 0.04 in English and 0.08 to 0.22 in Italian).
- **Speed** was measured on one desktop CPU (AMD Ryzen 9 9950X3D, ONNX Runtime 1.30) with Paradee's protocol.

## License

The weights in this repository are released under the Apache-2.0 license. They are trained on the outputs of
Kokoro-82M (Apache-2.0) with code derived from Paradee (Apache-2.0); `vocab.json` is Kokoro-82M's vocabulary.
The graphs take phoneme ids and contain no third-party software. Converting text to phonemes with the `titl`
package loads eSpeak NG and phonemizer-fork (both GPL-3.0-or-later) and num2words (LGPL-2.1), which pip installs
as separate packages; anyone redistributing them together with the package must follow their licenses. The
training sentences come from Wikipedia (CC BY-SA). Details: `THIRD_PARTY_NOTICES.md` in the code repository.

## Citation

```bibtex
@misc{angela2026teacher,
  author       = {Fabio Angela},
  title        = {Teacher in the Loop: Reliable Distillation of {Kokoro-82M} into {8M}-Parameter Single-Voice
                  Students, in {English} and {Italian}},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23400040},
  url          = {https://doi.org/10.5281/zenodo.23400040}
}

@misc{mahendrakar2026paradee,
  author       = {Sahil Mahendrakar},
  title        = {{Paradee}: Distilling {Kokoro-82M} into an {8M}-Parameter Single-Voice Text-to-Speech Model},
  year         = {2026},
  url          = {https://github.com/sahilmahendrakar/paradee}
}
```
