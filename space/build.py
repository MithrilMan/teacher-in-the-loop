"""Build the static Space: an in-browser demo of the released voices plus the paper's audio samples.

    python space/build.py SAMPLES_DIR ASSETS_DIR [OUT_DIR]   (default OUT_DIR: space/site; needs numpy, soundfile)

SAMPLES_DIR holds the rendered clips, one folder per voice and one per system inside it (held_<i>.flac): the
decoder-alone clips with a manifest.json per voice (rows with their texts), and the end-to-end clips with
e2e-manifest.json. They are not in this repository; the built Space holds them. Texts come from
e2e-manifest.json (eval/heldout.json); the build stops if the decoder-alone manifests name other sentences.

ASSETS_DIR holds what the demo runs on (space/README.md says how to get each one):
    weights/<voice>/model_int8.onnx, weights/SHA256SUMS   the int8 graphs of the weights repository, checked
                                                          against its checksum file
    lexicon/us_gold.json, us_silver.json                  Misaki 0.9.4's American lexicon (misaki/data/)
    espeak-ng/espeak-ng.js, espeak-ng.data, COPYING       @echogarden/espeak-ng-emscripten 0.3.5 cut to six
                                                          dictionaries by trim_espeak.py
The page code is in space/web/ and is copied as is; Kokoro's phoneme vocabulary comes from src/titl/vocab.json.
"""

import base64
import hashlib
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
CODE = "https://github.com/MithrilMan/teacher-in-the-loop"
WEIGHTS = "https://huggingface.co/MithrilMan/teacher-in-the-loop"
PAPER = CODE + "/blob/main/paper/Teacher-in-the-Loop-Angela-2026.pdf"
ORT_VERSION = "1.27.0"
ORT_WASM = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/ort-wasm-simd-threaded.wasm"
ORT_WASM_BYTES = 13479978
PEAKS = 56  # bars in each clip's waveform

DEMO_VOICES = {
    "af_heart": {"name": "Heart", "lang": "en", "meta": "English (US) · female", "orb": "heart"},
    "if_sara": {"name": "Sara", "lang": "it", "meta": "Italian · female", "orb": "sara"},
    "im_nicola": {"name": "Nicola", "lang": "it", "meta": "Italian · male", "orb": "nicola"},
}

# (folder, label, UTMOS over the 200 held-out sentences from the paper's tables, or None)
VOICES = {
    "af_heart": {
        "title": "af_heart, English",
        "alone": [("teacher", "Kokoro-82M, the teacher", "4.52"),
                  ("released", "Paradee, released decoder", "4.37"),
                  ("upstream", "Published recipe, our re-run (reproduction B)", "2.89"),
                  ("ours", "Teacher in the loop, final recipe", "4.47")],
        "e2e": [("released_e2e", "Paradee, released int8 graph (with its post-filter)"),
                ("ours_e2e", "Teacher in the loop, released int8 graph")],
    },
    "if_sara": {
        "title": "if_sara, Italian",
        "alone": [("teacher", "Kokoro-82M, the teacher", "4.02"),
                  ("upstream", "Published recipe, our run", "2.21"),
                  ("ours", "Teacher in the loop, final recipe", "3.88")],
        "e2e": [("ours_e2e", "Teacher in the loop, released int8 graph")],
    },
    "im_nicola": {
        "title": "im_nicola, Italian",
        "alone": [("teacher", "Kokoro-82M, the teacher", "3.87"),
                  ("upstream", "Published recipe, our run", "3.53"),
                  ("ours", "Teacher in the loop, final recipe", "3.70")],
        "e2e": [("ours_e2e", "Teacher in the loop, released int8 graph")],
    },
}

SPACE_README = f"""---
title: Teacher in the Loop
emoji: 🗣️
colorFrom: indigo
colorTo: green
sdk: static
pinned: false
license: apache-2.0
short_description: 8M-parameter Kokoro students, running in your browser
models:
- MithrilMan/teacher-in-the-loop
---

Demo and audio samples for the paper "Teacher in the Loop: Reliable Distillation of Kokoro-82M into 8M-Parameter
Single-Voice Students, in English and Italian" (Fabio Angela, 2026).

**Try it:** the three released voices run entirely in the browser. Text becomes phonemes through a JavaScript
port of Misaki's English rules and eSpeak NG compiled to WebAssembly; the voice's int8 ONNX graph runs on ONNX
Runtime Web, on one CPU thread. No server computes anything.

**Listen:** the Kokoro teacher, Paradee's released decoder, the published recipe as re-run by the author, and
the new recipe, on the first five held-out sentences of each voice.

Code: {CODE} (this Space is built from its `space/` folder). Weights: {WEIGHTS}.

## Third-party components

- `models/`: the int8 graphs of {WEIGHTS} (Apache-2.0), distilled from Kokoro-82M by hexgrad (Apache-2.0) with
  code derived from Paradee by Sahil Mahendrakar (Apache-2.0).
- `lexicon/`: the American English lexicon of Misaki 0.9.4 by hexgrad (Apache-2.0).
- `g2p-en.js`, `g2p-it.js`: JavaScript ports of Misaki 0.9.4's English G2P and eSpeak backend (Apache-2.0).
- `vendor/espeak-ng/`: eSpeak NG (GPL-3.0-or-later, see `vendor/espeak-ng/COPYING`), compiled to WebAssembly by
  the Echogarden project (@echogarden/espeak-ng-emscripten 0.3.5, source:
  https://github.com/echogarden-project/espeak-ng-emscripten; eSpeak NG source:
  https://github.com/espeak-ng/espeak-ng). The data package keeps only the en, it, es, fr, de and la
  dictionaries; `vendor/espeak-ng/trim_espeak.py` is the script that cut it. It runs as a separate program
  in its own module.
- ONNX Runtime Web {ORT_VERSION} (MIT), loaded from jsDelivr.
- Sample sentences: WikiText-103 and the Italian Wikipedia (CC BY-SA).
"""

ICONS = {
    "paper": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/></svg>',
    "code": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m8 7-5 5 5 5M16 7l5 5-5 5"/></svg>',
    "weights": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 3 8l9 5 9-5z"/><path d="m3 13 9 5 9-5"/></svg>',
    "chip": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="7" y="7" width="10" height="10" rx="2"/><path d="M9 3v2M15 3v2M9 19v2M15 19v2M3 9h2M3 15h2M19 9h2M19 15h2"/></svg>',
    "wave": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2"/></svg>',
    "info": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/></svg>',
    "download": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11m0 0-4.5-4.5M12 15l4.5-4.5M5 19h14"/></svg>',
}


def peaks(path: Path) -> tuple[str, float]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = np.abs(audio.mean(axis=1))
    bins = np.array_split(audio, PEAKS)
    p = np.array([b.max() if len(b) else 0.0 for b in bins])
    p = (p / max(p.max(), 1e-6)) ** 0.7
    return base64.b64encode(np.round(p * 255).astype(np.uint8).tobytes()).decode(), len(audio) / sr


def clip_html(out: Path, src: str, label: str, mos, ours: bool) -> str:
    pk, dur = peaks(out / src)
    mos_html = f'<span class="mos">UTMOS {mos}</span>' if mos else ""
    cls = "clip ours" if ours else "clip"
    return (f'<button type="button" class="{cls}" data-src="{src}" data-peaks="{pk}" data-dur="{dur:.3f}" '
            f'aria-label="Play: {html.escape(label)}"><span class="pb"></span><span class="lbl"><span>'
            f'{html.escape(label)}</span>{mos_html}</span><canvas></canvas></button>')


def copy_assets(assets: Path, out: Path) -> dict:
    sums = {}
    for line in (assets / "weights" / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.strip().lstrip("*")] = digest
    voices = {}
    (out / "models").mkdir()
    for v, spec in DEMO_VOICES.items():
        src = assets / "weights" / v / "model_int8.onnx"
        data = src.read_bytes()
        if hashlib.sha256(data).hexdigest() != sums[f"{v}/model_int8.onnx"]:
            sys.exit(f"{src}: not the int8 graph listed in SHA256SUMS")
        shutil.copy2(src, out / "models" / f"{v}.onnx")
        voices[v] = {"name": spec["name"], "lang": spec["lang"], "model": f"models/{v}.onnx", "bytes": len(data)}
    shutil.copytree(assets / "lexicon", out / "lexicon")
    (out / "vendor").mkdir()
    shutil.copytree(assets / "espeak-ng", out / "vendor" / "espeak-ng")
    shutil.copy2(HERE / "trim_espeak.py", out / "vendor" / "espeak-ng" / "trim_espeak.py")
    lex = {k: {"url": f"lexicon/us_{k}.json", "bytes": (assets / "lexicon" / f"us_{k}.json").stat().st_size}
           for k in ("gold", "silver")}
    return {
        "ort": {"wasm": ORT_WASM, "bytes": ORT_WASM_BYTES},
        "espeak": {"js": "vendor/espeak-ng/espeak-ng.js", "data": "vendor/espeak-ng/espeak-ng.data",
                   "bytes": (assets / "espeak-ng" / "espeak-ng.data").stat().st_size},
        "lexicon": lex,
        "voices": voices,
        "vocab": json.loads((HERE.parent / "src" / "titl" / "vocab.json").read_text(encoding="utf-8")),
    }


def stamp(out: Path, config_js: str) -> str:
    """Add a content version to the URLs of the page's own scripts, so that browsers do not keep an old copy
    after an update. Returns the version."""
    h = hashlib.sha256(config_js.encode())
    for f in sorted((HERE / "web").iterdir()):
        h.update(f.read_bytes())
    v = h.hexdigest()[:10]
    subs = {
        "app.js": [('from "./config.js"', f'from "./config.js?v={v}"'),
                   ("`./tts-worker.js${location.search}`",
                    f'`./tts-worker.js?v={v}${{location.search.replace("?", "&")}}`')],
        "tts-worker.js": [(f'from "./{m}"', f'from "./{m}?v={v}"') for m in ("config.js", "g2p-en.js", "g2p-it.js")],
    }
    for name, pairs in subs.items():
        text = (out / name).read_text(encoding="utf-8")
        for a, b in pairs:
            if text.count(a) != 1:
                sys.exit(f"{name}: expected one {a!r}")
            text = text.replace(a, b)
        (out / name).write_text(text, encoding="utf-8")
    return v


def main() -> None:
    src, assets = Path(sys.argv[1]), Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "site"
    e2e = json.loads((src / "e2e-manifest.json").read_text(encoding="utf-8"))
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(HERE / "web", out)
    config = copy_assets(assets, out)
    config_js = "// written by build.py\nexport const CONFIG = " + json.dumps(config, ensure_ascii=False) + ";\n"
    (out / "config.js").write_text(config_js, encoding="utf-8")
    version = stamp(out, config_js)
    (out / "audio").mkdir()

    tabs, panels = [], []
    for k, (voice, spec) in enumerate(VOICES.items()):
        texts = e2e[f"{voice}/ours_e2e"]["texts"]
        lab_texts = [r["text"] for r in json.loads((src / voice / "manifest.json").read_text(encoding="utf-8"))["rows"]]
        if [t.strip() for t in lab_texts] != [t.strip() for t in texts]:
            sys.exit(f"{voice}: the decoder-alone rows are not the held-out sentences of eval/heldout.json")
        for s in [s for s, *_ in spec["alone"]] + [s for s, _ in spec["e2e"]]:
            shutil.copytree(src / voice / s, out / "audio" / voice / s)
        blocks = []
        for i, text in enumerate(texts):
            alone = "".join(clip_html(out, f"audio/{voice}/{s}/held_{i}.flac", label, mos, s == "ours")
                            for s, label, mos in spec["alone"])
            e2e_html = "".join(clip_html(out, f"audio/{voice}/{s}/held_{i}.flac", label, None, s == "ours_e2e")
                               for s, label in spec["e2e"])
            blocks.append(f'<article class="sent"><p class="text"><span class="n">{i + 1}</span>{html.escape(text)}</p>'
                          f'<div class="group">Decoder alone, on the teacher\'s features</div><div class="clips">{alone}</div>'
                          f'<div class="group">End to end, from phonemes</div><div class="clips">{e2e_html}</div></article>')
        d = DEMO_VOICES[voice]
        sel = "true" if k == 0 else "false"
        tabs.append(f'<button type="button" class="tab" role="tab" aria-selected="{sel}" data-panel="panel-{voice}">'
                    f'<span class="orb {d["orb"]}"></span>{html.escape(spec["title"])}</button>')
        panels.append(f'<div class="panel" id="panel-{voice}" role="tabpanel"{"" if k == 0 else " hidden"}>{"".join(blocks)}</div>')

    voice_buttons = "".join(
        f'<button type="button" class="voice" role="radio" aria-checked="{"true" if k == 0 else "false"}" '
        f'data-voice="{v}"><span class="orb {d["orb"]}"></span><span><span class="vname">{d["name"]}</span>'
        f'<span class="vmeta">{d["meta"]} · {v}</span></span></button>'
        for k, (v, d) in enumerate(DEMO_VOICES.items()))
    model_mb = config["voices"]["af_heart"]["bytes"] / 1e6

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Teacher in the Loop</title>
<meta name="description" content="8M-parameter students of Kokoro-82M in English and Italian, running in your browser, with the audio samples of the paper Teacher in the Loop.">
<link rel="stylesheet" href="style.css?v={version}">
</head>
<body>
<div class="glow" aria-hidden="true"><i></i><i></i><i></i></div>
<header class="hero"><div class="wrap">
<p class="eyebrow">Fabio Angela · 2026 · Preprint</p>
<h1>Teacher in the Loop</h1>
<p class="lede">Kokoro-82M distilled into <strong>8M-parameter</strong> single-voice students, in English and
Italian. Type something below: the voice runs <strong>entirely in your browser</strong>.</p>
<nav class="pills"><a class="pill" href="{PAPER}">{ICONS["paper"]}Paper</a><a class="pill" href="{CODE}">{ICONS["code"]}Code</a><a class="pill" href="{WEIGHTS}">{ICONS["weights"]}Weights</a><a class="pill" href="#listen">{ICONS["wave"]}Samples</a></nav>
<ul class="facts">
<li><b>8M</b><span>parameters</span></li>
<li><b>{model_mb:.1f} MB</b><span>int8 graph</span></li>
<li><b>24 kHz</b><span>audio</span></li>
<li><b>100%</b><span>in your browser</span></li>
</ul>
</div></header>
<main class="wrap">
<section id="try" class="demo" aria-label="Try the voices">
<div class="demo-head"><h2>Try it</h2><span class="badge">{ICONS["chip"]}Runs on your device</span></div>
<div class="voices" role="radiogroup" aria-label="Voice">{voice_buttons}</div>
<div class="editor">
<textarea id="text" rows="3" spellcheck="false" aria-label="Text to speak"></textarea>
<div class="editor-foot"><div class="chips" id="examples"></div><span class="count" id="count"></span></div>
</div>
<div class="controls">
<label class="speed">Speed <input type="range" id="speed" min="0.5" max="2" step="0.05" value="1"><output id="speedVal">1.00×</output></label>
<span><kbd>Ctrl</kbd> + <kbd>Enter</kbd> &nbsp;<button type="button" id="speak" class="primary"></button></span>
</div>
<div class="loader" id="loader" hidden><div class="bar"><i></i></div><span id="loaderText"></span></div>
<div class="error" id="error" hidden></div>
<div class="player" id="player" hidden>
<button type="button" id="play" class="round" aria-label="Play or pause"></button>
<div class="wave" title="Click to seek"><canvas id="wave"></canvas></div>
<span class="time" id="time">0:00</span>
<button type="button" id="download" class="ghost" aria-label="Download WAV" title="Download WAV" disabled>{ICONS["download"]}</button>
</div>
<div class="stats" id="stats"></div>
<details class="phon" id="phonBox"><summary>Phonemes</summary><div id="phonemes"></div></details>
<div class="note">{ICONS["info"]}<span><strong>Not exactly the Python package.</strong> The voices are the
released int8 graphs, unchanged, but here the text is turned into phonemes by a JavaScript port of the pipeline
the paper uses. Italian gives the same phonemes on all 2,000 test sentences. English, which has no part-of-speech
tagger in the browser, gives the same phonemes on 95% of them; the rest differ in a word or two (mostly a
noun/verb homograph or the direction of a quotation mark), so some sentences can sound slightly different from
the <code>titl</code> package and from the paper's evaluation.</span></div>
<p class="fine">Everything happens in this tab. Text becomes phonemes with Misaki's lexicon and eSpeak NG compiled
to WebAssembly; the voice's int8 ONNX graph (text side and decoder, {model_mb:.1f} MB) then runs on ONNX Runtime Web,
on one CPU thread. What you type is not sent anywhere. The first use downloads the runtime and the voice.</p>
</section>

<section class="listen" id="listen">
<h2>Listen to the paper's samples</h2>
<p class="section-sub">The first five of the 200 held-out sentences of each voice, in their fixed order, not chosen by
listening. Every system on the same sentence, side by side.</p>
<details class="about"><summary>What you are hearing</summary>
<p><strong>Decoder alone.</strong> Every decoder receives the teacher's own durations, pitch, energy and phoneme
features for the sentence and the same random excitation seed, so only the decoder differs. These clips are rows of
the paper's 200-sentence evaluation; the UTMOS next to each system is its mean over those 200 sentences.</p>
<p><strong>End to end.</strong> Phonemes in, waveform out, through the released int8 ONNX graph of each model (text
side and decoder), as the <code>titl</code> package plays them. Paradee released only an English voice.</p>
<p><strong>Sentences and audio.</strong> WikiText-103 for English, the Italian Wikipedia for Italian; no student saw
them in training. 24 kHz, 16-bit FLAC, not normalized or edited. Headphones help: the differences between the
stronger systems are small.</p>
</details>
<div class="tabs" role="tablist">{"".join(tabs)}</div>
<p class="tip">Tip: switching to another system while a clip plays keeps the position, for quick A/B listening.</p>
{"".join(panels)}
</section>

<footer>
<p>Kokoro-82M by hexgrad; Paradee by Sahil Mahendrakar (arXiv:2610.06817); Misaki by hexgrad. eSpeak NG (GPL-3.0)
compiled to WebAssembly by the Echogarden project; ONNX Runtime Web by Microsoft. Sample sentences: WikiText-103 and
Wikipedia, CC BY-SA. Details in this Space's README.</p>
<p><a href="{PAPER}">Paper</a> · <a href="{CODE}">Code</a> · <a href="{WEIGHTS}">Weights</a></p>
</footer>
</main>
<script type="module">
// a private Space authorizes each file through the page's query string, so pass it on
import(`./app.js?v={version}${{location.search.replace("?", "&")}}`);
</script>
</body>
</html>
"""
    # the Space shows the page in a frame, and GitHub and Hugging Face refuse to be framed: open links outside
    page = page.replace('<a class="pill" href="https://', '<a class="pill" target="_blank" rel="noopener" href="https://')
    page = page.replace('<a href="https://', '<a target="_blank" rel="noopener" href="https://')
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "README.md").write_text(SPACE_README, encoding="utf-8")
    n = sum(1 for _ in (out / "audio").rglob("*.flac"))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{out / 'index.html'}: {n} clips, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
