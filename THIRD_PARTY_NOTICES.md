# Third-party notices

This repository's code is licensed under Apache-2.0 (`LICENSE`); the sentence lists in `training/data/` and
`eval/heldout.json` are licensed under CC BY-SA 4.0 (`training/data/LICENSE.md`). The released weights
(`https://huggingface.co/MithrilMan/teacher-in-the-loop`) are licensed under Apache-2.0. Attribution for the code
derived from Paradee and for Kokoro-82M is in `NOTICE`.

The code, the weights and the data depend on the third-party components below. Except where stated (the
Paradee files listed in section 2, the English sentence list in section 6 and the Misaki ports in section 7),
none of them is copied into this repository or into the weights: they are downloaded or installed by the user. Each license is the one stated by
the component's own license file, model card or package metadata, at the version or revision given.

## 1. Teacher model

| component | license | where | used for |
|---|---|---|---|
| Kokoro-82M v1.0 by hexgrad: weights `kokoro-v1_0.pth`, `config.json` and the voice packs `voices/af_heart.pt`, `voices/if_sara.pt`, `voices/im_nicola.pt`, revision `f3ff3571791e39611d31c381e3a41a3af07b4987` | Apache-2.0 (model card) | https://huggingface.co/hexgrad/Kokoro-82M | the teacher: the students are trained on its outputs, and `training/` and `eval/` run it; downloaded at the pinned revision, not redistributed. `src/titl/vocab.json` is the phoneme vocabulary from its `config.json`. |
| `kokoro` Python package 0.9.4 by hexgrad | Apache-2.0 | https://github.com/hexgrad/kokoro | imported by `training/` and `eval/` (Kokoro's model classes) |

## 2. Code derived from Paradee

| component | license | where |
|---|---|---|
| Paradee by Sahil Mahendrakar, training code at commit `9c8b4d7504cbee7e64de2d0341bb690f0b0ab708` | Apache-2.0 (`LICENSE` of that commit; upstream has no NOTICE file) | https://github.com/sahilmahendrakar/paradee |

`training/student.py` and `training/discriminators.py` are unchanged copies of upstream's `training/scripts/`
files; `training/student_decoder.py`, `training/gen_teacher.py`, `training/train_text.py`,
`training/train_decoder.py`, `training/export.py` and `src/titl/voice.py` are derived from upstream files. The
list of files and of the changes is in `NOTICE` and `training/README.md`. `training/data/wikitext_sents.json` is
Paradee's sentence list, unchanged (section 6).

## 3. Text to phonemes (runtime dependencies of the `titl` package)

Text is converted to phonemes as Kokoro does: Misaki 0.9.4 for English, Misaki with eSpeak NG for Italian, with
the dental stops t̪ and d̪ mapped to plain t and d.

| component | version | license | where |
|---|---|---|---|
| Misaki (hexgrad) | 0.9.4 | Apache-2.0 | https://github.com/hexgrad/misaki |
| eSpeak NG | 1.52.0, as shipped in the `espeakng-loader` 0.2.4 wheel | GPL-3.0-or-later | https://github.com/espeak-ng/espeak-ng |
| espeakng-loader (thewh1teagle): a wheel that ships the eSpeak NG shared library and its `espeak-ng-data` | 0.2.4 | MIT (repository); the wheel carries no license metadata, and the eSpeak NG library and data it contains are GPL-3.0-or-later | https://github.com/thewh1teagle/espeakng-loader |
| phonemizer-fork | 3.3.2 | GPL-3.0-or-later | https://pypi.org/project/phonemizer-fork/ (upstream https://github.com/bootphon/phonemizer) |
| num2words | 0.5.14 | LGPL-2.1 | https://github.com/savoirfairelinux/num2words |
| spaCy | 3.8.16 | MIT | https://github.com/explosion/spaCy |
| spaCy model `en_core_web_sm` (downloaded by Misaki on first use if missing) | 3.8.0 | MIT; its `LICENSES_SOURCES` lists the training sources (OntoNotes 5, licensed by Explosion; WordNet 3.0 License; ClearNLP conversion) | https://github.com/explosion/spacy-models |

**Copyleft at run time.** The `titl` package (Apache-2.0) does not contain eSpeak NG, phonemizer-fork or
num2words; pip installs them as separate packages. They are, however, loaded into the same process when text is
converted to phonemes: eSpeak NG as a shared library, phonemizer-fork and num2words as Python modules. This
holds for English too: Misaki's English pipeline also uses eSpeak NG. Using the package this way does not
change the license of this repository's code. Anyone who redistributes eSpeak NG, phonemizer-fork or num2words
together with this package (for example in a container image, a frozen executable or an application bundle)
must meet the GPL-3.0-or-later and LGPL-2.1 terms for those components, such as providing their license texts
and corresponding source. The ONNX graphs take phoneme ids and contain none of these components.

## 4. Python dependencies

Direct dependencies declared in `pyproject.toml`:

| package | extra | version in the paper's environment | license | where |
|---|---|---|---|---|
| numpy | base | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org |
| onnxruntime | base | 1.30.0 | MIT (ships its own `ThirdPartyNotices.txt`) | https://github.com/microsoft/onnxruntime |
| huggingface-hub | base | 1.31.0 | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| misaki | base | 0.9.4 | Apache-2.0 | https://github.com/hexgrad/misaki |
| spacy | base | 3.8.16 | MIT | https://github.com/explosion/spaCy |
| num2words | base | 0.5.14 | LGPL-2.1 | https://github.com/savoirfairelinux/num2words |
| phonemizer-fork | base | 3.3.2 | GPL-3.0-or-later | https://pypi.org/project/phonemizer-fork/ |
| espeakng-loader | base | 0.2.4 | no license metadata; MIT repository; bundles eSpeak NG (GPL-3.0-or-later) | https://github.com/thewh1teagle/espeakng-loader |
| torch | train | 2.14.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT (metadata) | https://github.com/pytorch/pytorch |
| torchaudio | train | 2.11.0 | BSD | https://github.com/pytorch/audio |
| kokoro | train | 0.9.4 | Apache-2.0 | https://github.com/hexgrad/kokoro |
| onnx | train | 1.22.0 | Apache-2.0 | https://github.com/onnx/onnx |
| transformers | eval | 5.17.0 | Apache-2.0 | https://github.com/huggingface/transformers |
| scipy | eval | 1.18.1 | BSD-3-Clause | https://github.com/scipy/scipy |
| pytest | test | (not in the training environment) | MIT | https://github.com/pytest-dev/pytest |

On Linux, torch 2.14.0 also installs NVIDIA's CUDA libraries (`nvidia-*`, `cuda-toolkit`), which are distributed
under NVIDIA's own license terms, not under an open-source license. Neither this repository nor the weights
contain them.

All packages of the paper's training and evaluation environment, with the license stated in their metadata
("base": installed by `pip install teacher-in-the-loop`; "train, eval": only with those extras). The exact
versions are pinned in `training/requirements-lock.txt`; a fresh inference install resolves to newer versions.

| package | version (paper environment) | installed with | license (package metadata) |
|---|---|---|---|
| addict | 2.4.0 | base | MIT (classifier) |
| annotated-doc | 0.0.5 | base | MIT |
| annotated-types | 0.8.0 | base | MIT |
| anyio | 4.15.1 | base | MIT |
| attrs | 26.1.0 | base | MIT |
| babel | 2.18.0 | base | BSD-3-Clause |
| blis | 1.3.3 | base | BSD |
| catalogue | 2.0.10 | base | MIT |
| certifi | 2026.7.22 | base | MPL-2.0 |
| charset-normalizer | 3.5.1 | base | MIT |
| click | 8.5.0 | base | BSD-3-Clause |
| cloudpathlib | 0.25.0 | base | MIT (classifier) |
| cloudpickle | 3.1.2 | base | BSD-3-Clause |
| confection | 1.3.3 | base | MIT |
| csvw | 4.1.0 | base | Apache-2.0 |
| cuda-bindings | 13.4.1 | train, eval | Apache-2.0 |
| cuda-pathfinder | 1.8.1 | train, eval | Apache-2.0 |
| cuda-toolkit | 13.0.3.0 | train, eval | no license metadata (NVIDIA CUDA Toolkit) |
| curated-tokenizers | 0.0.10 | train, eval | MIT |
| curated-transformers | 0.1.1 | train, eval | MIT |
| cymem | 2.0.13 | base | MIT |
| dlinfo | 2.0.0 | base | MIT |
| docopt | 0.6.2 | base | MIT |
| espeakng-loader | 0.2.4 | base | no license metadata; bundles eSpeak NG 1.52.0 (GPL-3.0-or-later), see above |
| filelock | 3.32.6 | base | MIT |
| flatbuffers | 25.12.19 | base | Apache-2.0 |
| fsspec | 2026.6.0 | base | BSD-3-Clause |
| h11 | 0.16.0 | base | MIT |
| hf-xet | 1.6.0 | base | Apache-2.0 |
| httpcore | 1.0.9 | base | BSD-3-Clause |
| httpx | 0.28.1 | base | BSD-3-Clause |
| huggingface_hub | 1.31.0 | base | Apache-2.0 |
| idna | 3.19 | base | BSD-3-Clause |
| isodate | 0.7.2 | base | BSD (classifier) |
| Jinja2 | 3.1.6 | base | BSD (classifier) |
| joblib | 1.6.0 | base | BSD-3-Clause |
| jsonschema | 4.26.0 | base | MIT |
| jsonschema-specifications | 2025.9.1 | base | MIT |
| kokoro | 0.9.4 | train, eval | Apache-2.0 |
| language-tags | 1.3.1 | base | MIT |
| loguru | 0.7.3 | train, eval | MIT (classifier) |
| markdown-it-py | 4.2.0 | base | MIT (classifier) |
| MarkupSafe | 3.0.3 | base | BSD-3-Clause |
| mdurl | 0.1.2 | base | MIT (classifier) |
| misaki | 0.9.4 | base | Apache-2.0 |
| ml_dtypes | 0.6.0 | train, eval | Apache-2.0 |
| mpmath | 1.3.0 | train, eval | BSD |
| murmurhash | 1.0.15 | base | MIT |
| networkx | 3.6.1 | train, eval | BSD-3-Clause |
| num2words | 0.5.14 | base | LGPL-2.1 (COPYING; metadata: LGPL) |
| numpy | 2.5.3 | base | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| nvidia-cublas | 13.1.1.3 | train, eval | LicenseRef-NVIDIA-Proprietary |
| nvidia-cuda-cupti | 13.0.85 | train, eval | LicenseRef-NVIDIA-Proprietary |
| nvidia-cuda-nvrtc | 13.0.88 | train, eval | LicenseRef-NVIDIA-Proprietary |
| nvidia-cuda-runtime | 13.0.96 | train, eval | LicenseRef-NVIDIA-Proprietary |
| nvidia-cudnn-cu13 | 9.24.0.43 | train, eval | LicenseRef-NVIDIA-Proprietary |
| nvidia-cufft | 12.0.0.61 | train, eval | LicenseRef-NVIDIA-Proprietary |
| nvidia-cufile | 1.15.1.6 | train, eval | LicenseRef-NVIDIA-Proprietary |
| nvidia-curand | 10.4.0.35 | train, eval | LicenseRef-NVIDIA-Proprietary |
| nvidia-cusolver | 12.0.4.66 | train, eval | LicenseRef-NVIDIA-Proprietary |
| nvidia-cusparse | 12.6.3.3 | train, eval | LicenseRef-NVIDIA-Proprietary |
| nvidia-cusparselt-cu13 | 0.8.1 | train, eval | LicenseRef-NVIDIA-Proprietary |
| nvidia-nccl-cu13 | 2.30.7 | train, eval | LicenseRef-NVIDIA-Proprietary |
| nvidia-nvjitlink | 13.4.52 | train, eval | LicenseRef-NVIDIA-Proprietary |
| nvidia-nvshmem-cu13 | 3.4.5 | train, eval | LicenseRef-NVIDIA-Proprietary |
| nvidia-nvtx | 13.0.85 | train, eval | Apache-2.0 |
| onnx | 1.22.0 | train, eval | Apache-2.0 |
| onnxruntime | 1.30.0 | base | MIT |
| packaging | 26.3 | base | Apache-2.0 OR BSD-2-Clause |
| phonemizer-fork | 3.3.2 | base | GPL-3.0-or-later |
| preshed | 3.0.13 | base | MIT |
| protobuf | 7.36.1 | base | BSD-3-Clause |
| pydantic | 2.13.5 | base | MIT |
| pydantic_core | 2.46.5 | base | MIT |
| Pygments | 2.21.0 | base | BSD-2-Clause |
| pyparsing | 3.3.2 | base | MIT |
| python-dateutil | 2.9.0.post0 | base | Apache-2.0 or BSD-3-Clause (dual, classifiers) |
| PyYAML | 6.0.3 | base | MIT |
| rdflib | 7.6.0 | base | BSD-3-Clause |
| referencing | 0.37.0 | base | MIT |
| regex | 2026.9.10 | base | Apache-2.0 AND CNRI-Python |
| requests | 2.34.2 | base | Apache-2.0 |
| rfc3986 | 1.5.0 | base | Apache-2.0 |
| rich | 15.0.0 | base | MIT |
| rpds-py | 2026.6.3 | base | MIT |
| safetensors | 0.8.0 | train, eval | Apache-2.0 (classifier) |
| scipy | 1.18.1 | train, eval | BSD-3-Clause (license text) |
| segments | 2.4.0 | base | Apache-2.0 |
| setuptools | 84.0.0 | base | MIT |
| shellingham | 1.5.4 | base | ISC |
| six | 1.17.0 | base | MIT |
| smart_open | 8.0.1 | base | MIT (classifier) |
| spacy | 3.8.16 | base | MIT |
| spacy-curated-transformers | 0.3.1 | train, eval | MIT |
| spacy-legacy | 3.0.12 | base | MIT |
| spacy-loggers | 1.0.5 | base | MIT |
| srsly | 2.5.3 | base | MIT |
| sympy | 1.14.0 | train, eval | BSD |
| termcolor | 3.3.0 | base | MIT |
| thinc | 8.3.13 | base | MIT |
| tokenizers | 0.23.2 | train, eval | Apache-2.0 (classifier) |
| torch | 2.14.0 | train, eval | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| torchaudio | 2.11.0 | train, eval | BSD (classifier) |
| tqdm | 4.70.1 | base | MPL-2.0 AND MIT |
| transformers | 5.17.0 | train, eval | Apache-2.0 |
| triton | 3.8.0 | train, eval | MIT |
| typer | 0.27.2 | base | MIT |
| typing_extensions | 4.16.0 | base | PSF-2.0 |
| typing-inspection | 0.4.4 | base | MIT |
| uritemplate | 4.2.0 | base | BSD-3-Clause OR Apache-2.0 |
| urllib3 | 2.7.0 | base | MIT |
| wasabi | 1.1.3 | base | MIT |
| weasel | 1.0.0 | base | MIT |
| wrapt | 2.4.1 | base | BSD-2-Clause |

## 5. Models used only for evaluation

None of these is redistributed; `eval/` and `training/utmos.py` download them at run time.

| component | license | where | used for |
|---|---|---|---|
| SpeechMOS by tarepan, commit `ed25eacbfa42b99156c36ebec67a733b5dbb9b79` | MIT | https://github.com/tarepan/SpeechMOS | UTMOS implementation (`training/utmos.py`, loaded with `torch.hub`) |
| UTMOS checkpoint `utmos22_strong_step7459_v1.pt` (SHA-256 `38aa51ab79e2a4e09a1449758a4b37e9cbb2e8235a49662a732d33a9ba1e9bff`), downloaded from SpeechMOS release v1.0.0 | MIT: SpeechMOS states that it is derived from the UTMOS22 "strong learner" checkpoint, MIT, Copyright 2022 Saruwatari & Koyama laboratory, The University of Tokyo; its wav2vec 2.0 model code comes from fairseq (MIT) | https://github.com/sarulab-speech/UTMOS22 | the paper's main metric (UTMOS, Saeki et al., Interspeech 2022) |
| Whisper large-v3-turbo (`openai/whisper-large-v3-turbo`; the paper used revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`) | MIT (model card) | https://huggingface.co/openai/whisper-large-v3-turbo | word error rate (`eval/wer_similarity.py`) |
| WavLM base plus, speaker verification (`microsoft/wavlm-base-plus-sv`; the paper used revision `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`) | the model card refers to the license of the UniSpeech repository, CC BY-SA 3.0 | https://huggingface.co/microsoft/wavlm-base-plus-sv, https://github.com/microsoft/UniSpeech | speaker similarity (`eval/wer_similarity.py`) |
| Paradee-8M-v1.0, released model (`sahilmahendrakar/Paradee-8M-v1.0`, revision `8f34b01ef8adcb0bec470b89bf2fd62a7b2369e6`) | Apache-2.0 (model card) | https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.0 | the "released decoder" baseline in the paper's tables and the speed comparison (`eval/bench_onnx.py`) |

## 6. Data

| data | license | where | used for |
|---|---|---|---|
| WikiText-103 (Merity, Xiong, Bradbury and Socher, "Pointer Sentinel Mixture Models", arXiv:1609.07843), from English Wikipedia | CC BY-SA 3.0 | https://arxiv.org/abs/1609.07843 | English sentences; `training/data/wikitext_sents.json` is Paradee's list of 12,000 sentences (same sentences, same order as Paradee commit `9c8b4d7`) |
| Italian Wikipedia, snapshot `20231101.it` of the Hugging Face dataset `wikimedia/wikipedia` (revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`) | CC BY-SA 3.0 and GFDL (dataset card) | https://huggingface.co/datasets/wikimedia/wikipedia | Italian sentences; `training/data/italian_sents.json` keeps the URL of each sentence's article |
| speech rendered by Kokoro-82M (section 1) | Apache-2.0 (the teacher's license) | regenerated by `training/gen_teacher.py` and `training/pool.py` | the students' training targets; no audio is redistributed |

The sentence lists in this repository are distributed under CC BY-SA 4.0, with attribution and changes in
`training/data/LICENSE.md`.

## 7. In-browser demo (`space/`)

`space/` builds the static Hugging Face Space (`https://huggingface.co/spaces/MithrilMan/teacher-in-the-loop`).
The repository holds only its code; the built site bundles the components marked "in the built site".

| component | version | license | where | used for |
|---|---|---|---|---|
| Misaki (hexgrad): `en.py` and `espeak.py` | 0.9.4 | Apache-2.0 | https://github.com/hexgrad/misaki | ported to JavaScript in `space/web/g2p-en.js` and `space/web/g2p-it.js` (derived code, see `NOTICE`) |
| Misaki's American English lexicon `us_gold.json`, `us_silver.json` | 0.9.4 | Apache-2.0 | https://github.com/hexgrad/misaki | in the built site (`lexicon/`), unchanged |
| eSpeak NG compiled to WebAssembly by the Echogarden project, `@echogarden/espeak-ng-emscripten` | 0.3.5 | GPL-3.0 (package metadata; eSpeak NG is GPL-3.0-or-later) | https://github.com/echogarden-project/espeak-ng-emscripten, https://github.com/espeak-ng/espeak-ng | in the built site (`vendor/espeak-ng/`), with its license, its source links and `trim_espeak.py`, which removes the dictionaries the demo does not use from the data package; a separate program loaded in its own module |
| ONNX Runtime Web | 1.27.0 | MIT | https://github.com/microsoft/onnxruntime | runs the int8 graphs in the browser; loaded from jsDelivr, not redistributed |
| the int8 graphs of the weights repository | revision pinned in `src/titl/voice.py` | Apache-2.0 | https://huggingface.co/MithrilMan/teacher-in-the-loop | in the built site (`models/`) |
