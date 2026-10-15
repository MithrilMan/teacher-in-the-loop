// Module worker: text -> Kokoro phonemes -> int8 ONNX graph -> 24 kHz audio, all on this device.
// Messages in:  {type: "load", voice}  |  {type: "speak", id, voice, text, speed}  |  {type: "cancel"}
// Messages out: progress, ready, chunk (one per sentence piece, audio transferred), done, error.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.wasm.min.mjs";
import { CONFIG } from "./config.js";
import { makeEnglish } from "./g2p-en.js";
import { makeItalian } from "./g2p-it.js";

ort.env.wasm.numThreads = 1;  // one thread is many times faster than real time and needs no special headers

const SENTENCE_BREAK = /(?<=[.!?…])\s+|\n+/;  // as titl's Phonemizer.chunks
const MAX_PHONEMES = 510;

let runtime = null;          // promise: eSpeak module + worker; also sets ONNX Runtime's wasm binary
let english = null, italian = null;
const voices = new Map();    // voice -> promise of {session, g2p}
let current = 0;             // id of the newest speak request; older ones stop at the next sentence
let queue = Promise.resolve();

const post = (msg, transfer = []) => self.postMessage(msg, transfer);

async function fetchBytes(url, bytes, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const reader = res.body.getReader();
  const parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    post({ type: "progress", label, loaded, total: Math.max(bytes, loaded) });
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

function loadRuntime() {
  runtime ??= (async () => {
    const [wasm, data] = await Promise.all([
      fetchBytes(CONFIG.ort.wasm, CONFIG.ort.bytes, "runtime"),
      fetchBytes(new URL(CONFIG.espeak.data, import.meta.url).href, CONFIG.espeak.bytes, "espeak"),
    ]);
    ort.env.wasm.wasmBinary = wasm;
    const { default: Module } = await import(new URL(CONFIG.espeak.js, import.meta.url).href);
    const m = await Module({ getPreloadedPackage: () => data.buffer });
    return { m, espeak: new m.eSpeakNGWorker() };
  })().catch(e => { runtime = null; throw e; });
  return runtime;
}

function loadG2P(lang, rt) {
  if (lang === "it") return (italian ??= Promise.resolve(makeItalian(rt.m, rt.espeak)));
  english ??= (async () => {
    const dec = new TextDecoder();
    const [gold, silver] = await Promise.all([
      fetchBytes(CONFIG.lexicon.gold.url, CONFIG.lexicon.gold.bytes, "lexicon-gold"),
      fetchBytes(CONFIG.lexicon.silver.url, CONFIG.lexicon.silver.bytes, "lexicon-silver"),
    ]);
    return makeEnglish(rt.m, rt.espeak, JSON.parse(dec.decode(gold)), JSON.parse(dec.decode(silver)));
  })().catch(e => { english = null; throw e; });
  return english;
}

function loadVoice(voice) {
  if (!voices.has(voice)) {
    const v = CONFIG.voices[voice];
    voices.set(voice, (async () => {
      const model = fetchBytes(v.model, v.bytes, `model-${voice}`);
      const rt = await loadRuntime();
      const [bytes, g2p] = await Promise.all([model, loadG2P(v.lang, rt)]);
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      return { session, g2p };
    })().catch(e => { voices.delete(voice); throw e; }));
  }
  return voices.get(voice);
}

// cut a phoneme string into pieces that fit the text side, at the last space that fits (titl.split_phonemes)
function splitPhonemes(ps) {
  const out = [];
  let rest = ps.trim();
  while (rest.length > MAX_PHONEMES) {
    let cut = rest.lastIndexOf(" ", MAX_PHONEMES);
    if (cut <= 0) cut = MAX_PHONEMES;
    out.push(rest.slice(0, cut).trimEnd());
    rest = rest.slice(cut).trimStart();
  }
  if (rest) out.push(rest);
  return out;
}

async function speak({ id, voice, text, speed }) {
  const t0 = performance.now();
  const { session, g2p } = await loadVoice(voice);
  post({ type: "ready", voice, ms: performance.now() - t0 });
  const sentences = text.trim().split(SENTENCE_BREAK).filter(s => s.trim());
  let index = 0;
  for (const sentence of sentences) {
    if (id !== current) return;
    const tg = performance.now();
    const pieces = splitPhonemes(g2p(sentence));
    const g2pMs = performance.now() - tg;
    for (const [k, piece] of pieces.entries()) {
      if (id !== current) return;
      const ids = [0];
      for (const c of piece) if (c in CONFIG.vocab) ids.push(CONFIG.vocab[c]);
      ids.push(0);
      const tr = performance.now();
      const out = await session.run({
        input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, x => BigInt(x)), [1, ids.length]),
        speed: new ort.Tensor("float32", new Float32Array([speed]), [1]),
      });
      const runMs = performance.now() - tr;
      const audio = new Float32Array(out.waveform.data);
      post({ type: "chunk", id, index: index++, text: k === 0 ? sentence : "", phonemes: piece, audio,
             g2pMs: k === 0 ? g2pMs : 0, runMs }, [audio.buffer]);
    }
  }
  post({ type: "done", id });
}

self.onmessage = ({ data: msg }) => {
  if (msg.type === "load") {
    loadVoice(msg.voice).then(() => post({ type: "ready", voice: msg.voice }),
                              e => post({ type: "error", message: String(e?.message ?? e) }));
  } else if (msg.type === "cancel") {
    current = -1;
  } else if (msg.type === "speak") {
    current = msg.id;
    queue = queue.then(() => (msg.id === current ? speak(msg) : null))
      .catch(e => post({ type: "error", id: msg.id, message: String(e?.message ?? e) }));
  }
};
