# space/: the Hugging Face Space

Source of https://huggingface.co/spaces/MithrilMan/teacher-in-the-loop, a static page with two parts:

- **Try it**: the three released voices run in the visitor's browser. Text becomes phonemes in JavaScript
  (Misaki's English rules and lexicon, eSpeak NG compiled to WebAssembly), and the voice's int8 ONNX graph runs
  on ONNX Runtime Web, on one CPU thread, in a Web Worker. Nothing is computed on a server.
- **Listen**: the paper's audio samples (teacher, Paradee's released decoder, the published recipe, ours) on
  the first five held-out sentences of each voice.

```
web/app.js          page logic: voice picker, streaming playback, waveform, statistics, sample players
web/tts-worker.js   Web Worker: downloads the runtime and a voice, text -> phonemes -> ids -> audio, one sentence at a time
web/g2p-en.js       English text -> Kokoro phonemes: a port of Misaki 0.9.4's en.G2P (American, eSpeak NG fallback)
web/g2p-it.js       Italian text -> Kokoro phonemes: a port of Misaki's eSpeak NG backend with the dental mapping
web/style.css       light and dark theme
build.py            writes the site: index.html, config.js, the Space's README, graphs, lexicon, eSpeak NG, samples
trim_espeak.py      cuts the eSpeak NG WebAssembly data package down to a few dictionaries
```

## Phonemes in the browser

The voices are the released graphs, unchanged; what differs from the `titl` package is the text-to-phoneme
step, rewritten in JavaScript. Against the Python pipeline on 2,000 sentences per language (held-out and
training sentences): Italian gives identical phonemes on 2,000 of 2,000. English gives identical phonemes on
1,898 of 2,000 sentences and 99.7% of words: without spaCy, part-of-speech tags are guessed from the
neighbouring words, so some noun/verb homographs and the direction of straight quotes can differ. The page
says so.

## Build

1. Assets, in one folder (`ASSETS`):
   - `weights/`: the int8 graphs and the checksum file of the weights repository, at the revision pinned in
     `src/titl/voice.py`:
     `hf download MithrilMan/teacher-in-the-loop --revision <REVISION> --include "*/model_int8.onnx" --include SHA256SUMS --local-dir ASSETS/weights`
   - `lexicon/`: `us_gold.json` and `us_silver.json` from the `misaki` 0.9.4 package (`misaki/data/`).
   - `espeak-ng/`: `npm pack @echogarden/espeak-ng-emscripten@0.3.5`, unpack the tarball, then
     `python space/trim_espeak.py <unpacked>/package ASSETS/espeak-ng en,it,es,fr,de,la` (24.0 MB -> 1.2 MB;
     Italian text switches to the es, fr, de and la dictionaries for some words).
2. `python space/build.py SAMPLES ASSETS` (needs `numpy` and `soundfile`; `SAMPLES` is the folder of
   rendered clips described in `build.py`). The site is written to `space/site/`.
3. Try it locally with `python -m http.server --directory space/site`, then upload:
   `hf upload MithrilMan/teacher-in-the-loop space/site . --repo-type space`.

ONNX Runtime Web is loaded from jsDelivr at a pinned version (`ORT_VERSION` in `build.py`). While the Space is
private, Hugging Face authorizes each file through the page's query string; the page passes it on to its
scripts and to the worker.

## Licenses

The code in this folder is Apache-2.0. `web/g2p-en.js` and `web/g2p-it.js` are derived from Misaki
(Apache-2.0, see `NOTICE`). The built site also contains Misaki's lexicon (Apache-2.0) and eSpeak NG
(GPL-3.0-or-later) as a separate WebAssembly program with its license and the script that trimmed it; see
`THIRD_PARTY_NOTICES.md`, section 7.
