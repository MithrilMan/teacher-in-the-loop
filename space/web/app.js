// Page logic: the in-browser demo (a worker synthesizes, this thread plays and draws) and the sample players.
import { CONFIG } from "./config.js";

const $ = s => document.querySelector(s);
const SR = 24000;
const BLOCK = 240;            // waveform peaks every 10 ms
const MAX_CHARS = 1500;
const ICON = {
  play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.5v13a1 1 0 0 0 1.5.87l11-6.5a1 1 0 0 0 0-1.74l-11-6.5A1 1 0 0 0 8 5.5z"/></svg>',
  pause: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4.5" height="14" rx="1.2"/><rect x="13.5" y="5" width="4.5" height="14" rx="1.2"/></svg>',
  stop: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2.5"/></svg>',
  speak: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2"/></svg>',
};
const EXAMPLES = {
  en: ["Eight million parameters, nine megabytes, and the whole voice runs right here, in your browser.",
       "On a quiet Sunday morning in 1923, the lighthouse keeper wrote his last letter home.",
       "Is this the teacher, or a student that learned by listening? Close your eyes and decide."],
  it: ["Otto milioni di parametri, e tutta la voce funziona qui, direttamente nel tuo browser.",
       "La nebbia saliva lenta dal fiume, e le campane della città suonavano le sette.",
       "Il treno per Milano partirà dal binario 4, con dieci minuti di ritardo. Ci scusiamo per il disagio."],
};
const DEFAULT_TEXT = { af_heart: EXAMPLES.en[0], if_sara: EXAMPLES.it[0], im_nicola: EXAMPLES.it[1] };

// ---------- colors from the CSS tokens ----------
let colors = {};
function readColors() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of ["a1", "a2", "wave", "faint"]) colors[k] = cs.getPropertyValue(`--${k}`).trim();
}
readColors();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); drawAllClips(); });

// ---------- audio context (created on the first click) ----------
let ctx = null;
function audioCtx() {
  ctx ??= new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === "suspended") ctx.resume();
  return ctx;
}

// ---------- demo ----------
const worker = new Worker(new URL(`./tts-worker.js${location.search}`, import.meta.url), { type: "module" });
const ui = {
  text: $("#text"), count: $("#count"), chips: $("#examples"), speed: $("#speed"), speedVal: $("#speedVal"),
  speak: $("#speak"), loader: $("#loader"), loaderBar: $("#loader i"), loaderText: $("#loaderText"),
  player: $("#player"), play: $("#play"), wave: $("#wave"), time: $("#time"), download: $("#download"),
  stats: $("#stats"), phonBox: $("#phonBox"), phonemes: $("#phonemes"), error: $("#error"),
};
let voice = "af_heart";
let nextId = 1;
let utt = null;
const ready = new Set();
const pending = new Set();      // voices being downloaded
const progress = new Map();     // asset label -> {loaded, total}
let level = 0;

const lang = v => CONFIG.voices[v].lang;
const fmt = samples => { const s = Math.max(0, samples / SR); return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; };

function setExamples() {
  ui.chips.replaceChildren(...EXAMPLES[lang(voice)].map(t => {
    const b = document.createElement("button");
    b.className = "chip"; b.type = "button"; b.textContent = t; b.title = t;
    b.onclick = () => { ui.text.value = t; updateCount(); speak(); };
    return b;
  }));
}
function updateCount() { ui.count.textContent = `${ui.text.value.length} / ${MAX_CHARS}`; }

function selectVoice(v) {
  const wasDefault = Object.values(EXAMPLES).flat().includes(ui.text.value.trim()) || !ui.text.value.trim();
  const langChanged = lang(v) !== lang(voice);
  voice = v;
  for (const b of document.querySelectorAll(".voice")) b.setAttribute("aria-checked", String(b.dataset.voice === v));
  if (wasDefault && (langChanged || !ui.text.value.trim())) { ui.text.value = DEFAULT_TEXT[v]; updateCount(); }
  setExamples();
  prefetch(v);
}

function prefetch(v) {
  if (ready.has(v) || pending.has(v)) return;
  pending.add(v);
  worker.postMessage({ type: "load", voice: v });
}

function showLoader() {
  if (!pending.size) { ui.loader.hidden = true; return; }
  let loaded = 0, total = 0;
  for (const p of progress.values()) { loaded += p.loaded; total += p.total; }
  const pct = total ? Math.min(100, (100 * loaded) / total) : 0;
  ui.loader.hidden = false;
  ui.loaderBar.style.width = `${pct}%`;
  const names = [...pending].map(v => CONFIG.voices[v].name).join(", ");
  ui.loaderText.textContent = `Downloading ${names} and the runtime: ${pct.toFixed(0)}%. Only the first time.`;
  if (utt && !utt.chunks.length && !utt.done) setSpeakButton("loading", pct);
}

function setSpeakButton(state, pct = 0) {
  ui.speak.classList.toggle("busy", state !== "idle");
  ui.speak.innerHTML = state === "idle" ? `${ICON.speak}<span>Speak</span>`
    : state === "loading" ? `${ICON.stop}<span>Loading ${pct.toFixed(0)}%</span>` : `${ICON.stop}<span>Stop</span>`;
}

function stopAudio() {
  if (!utt) return;
  for (const s of utt.sources) { try { s.stop(); } catch { /* already stopped */ } }
  utt.sources = [];
}

function speak() {
  const text = ui.text.value.trim().slice(0, MAX_CHARS);
  if (!text) { ui.text.focus(); return; }
  if (utt && !utt.done) { cancel(); return; }
  pauseClip();
  audioCtx();
  stopAudio();
  ui.error.hidden = true;
  const speed = +ui.speed.value;
  const id = nextId++;
  const needsLoad = !ready.has(voice);
  utt = { id, voice, chunks: [], samples: 0, peaks: [], maxPeak: 0.05, segs: [], sources: [], next: 0, done: false,
          playing: true, pausedAt: 0, t0: performance.now(), firstAt: null, compute: 0, needsLoad, loadMs: 0,
          estimate: (text.length / 14 / speed) * SR, scale: 0, full: null, rows: [] };
  utt.scale = utt.estimate;
  if (needsLoad) pending.add(voice);
  setSpeakButton(needsLoad ? "loading" : "busy");
  ui.player.hidden = false;
  ui.download.disabled = true;
  ui.play.innerHTML = ICON.pause;
  ui.stats.replaceChildren();
  ui.phonemes.replaceChildren();
  worker.postMessage({ type: "speak", id, voice, text, speed });
  loop();
}

function cancel() {
  worker.postMessage({ type: "cancel" });
  if (utt) { finish(utt, true); stopAudio(); utt.playing = false; utt.pausedAt = 0; ui.play.innerHTML = ICON.play; }
}

function finish(u, cancelled = false) {
  if (u.done) return;
  u.done = true;
  setSpeakButton("idle");
  ui.download.disabled = !u.samples;
  if (!u.samples) { ui.player.hidden = true; return; }
  const secs = u.samples / SR, compute = u.compute / 1000;
  const stat = (html, hot) => { const s = document.createElement("span"); s.className = `stat${hot ? " hot" : ""}`; s.innerHTML = html; return s; };
  ui.stats.replaceChildren(
    stat(`<b>${secs.toFixed(1)} s</b> of audio computed in <b>${compute.toFixed(2)} s</b>`),
    stat(`<b>${(secs / compute).toFixed(1)}×</b> faster than real time`, true),
    stat(`first sound after <b>${(u.firstAt / 1000).toFixed(2)} s</b>${u.needsLoad ? " (with the download)" : ""}`),
    stat("this device · 1 CPU thread · int8"),
    ...(cancelled ? [stat("stopped early")] : []),
  );
}

function schedule(u, audio, offset) {
  const c = audioCtx();
  const buf = c.createBuffer(1, audio.length, SR);
  buf.copyToChannel(audio, 0);
  const src = c.createBufferSource();
  src.buffer = buf;
  src.connect(c.destination);
  const when = Math.max(c.currentTime + 0.04, u.next);
  src.start(when);
  u.sources.push(src);
  u.segs.push({ when, offset, length: audio.length });
  u.next = when + audio.length / SR;
}

function fullAudio(u) {
  if (u.full && u.full.length === u.samples) return u.full;
  const out = new Float32Array(u.samples);
  let o = 0;
  for (const c of u.chunks) { out.set(c, o); o += c.length; }
  return (u.full = out);
}

function playFrom(pos) {
  const u = utt;
  if (!u || !u.samples) return;
  pauseClip();
  stopAudio();
  u.segs = [];
  u.next = 0;
  pos = Math.max(0, Math.min(Math.floor(pos), u.samples - 1));
  schedule(u, fullAudio(u).subarray(pos), pos);
  u.playing = true;
  ui.play.innerHTML = ICON.pause;
  loop();
}

function pauseMain() {
  if (!utt || !utt.playing) return;
  utt.pausedAt = currentPos();
  utt.playing = false;
  stopAudio();
  ui.play.innerHTML = ICON.play;
}

function currentPos() {
  const u = utt;
  if (!u) return 0;
  if (!u.playing || !ctx || !u.segs.length) return u.pausedAt;
  const t = ctx.currentTime;
  let pos = u.segs[0].offset;
  for (const s of u.segs) {
    if (t < s.when) break;
    pos = Math.min(s.offset + (t - s.when) * SR, s.offset + s.length);
  }
  return pos;
}

function showError(message) {
  pending.clear();
  progress.clear();
  showLoader();
  ui.error.hidden = false;
  ui.error.textContent = `Something went wrong: ${message}`;
  if (utt && !utt.done) finish(utt, true);
  setSpeakButton("idle");
}
worker.onerror = e => { e.preventDefault(); showError(e.message || "the speech worker could not start"); };

worker.onmessage = ({ data: m }) => {
  if (m.type === "progress") {
    progress.set(m.label, { loaded: m.loaded, total: m.total });
    showLoader();
  } else if (m.type === "ready") {
    ready.add(m.voice);
    pending.delete(m.voice);
    if (!pending.size) progress.clear();
    showLoader();
    if (utt && utt.voice === m.voice && !utt.done && !utt.chunks.length) setSpeakButton("busy");
  } else if (m.type === "chunk") {
    const u = utt;
    if (!u || m.id !== u.id || u.done) return;
    const offset = u.samples;
    u.chunks.push(m.audio);
    u.samples += m.audio.length;
    u.compute += m.g2pMs + m.runMs;
    for (let i = 0; i < m.audio.length; i += BLOCK) {
      let p = 0;
      for (let j = i, e = Math.min(i + BLOCK, m.audio.length); j < e; j++) { const a = Math.abs(m.audio[j]); if (a > p) p = a; }
      u.peaks.push(p);
      if (p > u.maxPeak) u.maxPeak = p;
    }
    if (u.firstAt === null) { u.firstAt = performance.now() - u.t0; setSpeakButton("busy"); }
    if (u.playing) schedule(u, m.audio, offset);
    addPhonemes(m.text, m.phonemes);
  } else if (m.type === "done") {
    if (utt && m.id === utt.id) finish(utt);
  } else if (m.type === "error") {
    showError(m.message);
  }
};

function addPhonemes(text, ps) {
  const row = document.createElement("div");
  row.className = "phon-row";
  if (text) { const s = document.createElement("div"); s.className = "src"; s.textContent = text; row.append(s); }
  const p = document.createElement("div");
  p.className = "ps"; p.textContent = ps;
  row.append(p);
  ui.phonemes.append(row);
}

// ---------- drawing ----------
function sizeCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return [g, w, h];
}

function bars(g, w, h, n, value, playedUpTo, opts = {}) {
  const step = w / n, bw = Math.max(1.5, step * 0.62);
  const grad = g.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, colors.a1);
  grad.addColorStop(1, colors.a2);
  for (let i = 0; i < n; i++) {
    const v = value(i);
    if (v === null) {
      if (opts.pendingDots) {
        g.globalAlpha = 0.35 + 0.3 * Math.sin(performance.now() / 260 - i * 0.18);
        g.fillStyle = colors.wave;
        g.fillRect(i * step, h / 2 - 1, bw, 2);
        g.globalAlpha = 1;
      }
      continue;
    }
    const bh = Math.max(2, v * (h - 2));
    g.fillStyle = (i + 0.5) / n <= playedUpTo ? grad : colors.wave;
    g.beginPath();
    g.roundRect ? g.roundRect(i * step, (h - bh) / 2, bw, bh, bw / 2) : g.rect(i * step, (h - bh) / 2, bw, bh);
    g.fill();
  }
}

function drawMain() {
  const u = utt;
  if (!u || ui.player.hidden) return;
  const [g, w, h] = sizeCanvas(ui.wave);
  const target = u.done ? u.samples : Math.max(u.samples * 1.04, u.estimate);
  u.scale += (target - u.scale) * 0.12;
  const n = Math.max(24, Math.floor(w / 4.5));
  const pos = currentPos();
  bars(g, w, h, n, i => {
    const s0 = (i * u.scale) / n, s1 = ((i + 1) * u.scale) / n;
    if (s0 >= u.samples) return null;
    let p = 0;
    for (let b = Math.floor(s0 / BLOCK), e = Math.min(u.peaks.length, Math.ceil(s1 / BLOCK)); b < e; b++) if (u.peaks[b] > p) p = u.peaks[b];
    return Math.pow(p / u.maxPeak, 0.7);
  }, pos / u.scale, { pendingDots: !u.done });
  ui.time.textContent = `${fmt(pos)} / ${fmt(u.samples)}${u.done ? "" : " …"}`;
  // the voice's orb follows the loudness at the playhead
  const b = Math.floor(pos / BLOCK);
  const target_level = u.playing && b < u.peaks.length ? Math.min(1, u.peaks[b] / u.maxPeak) : 0;
  level += (target_level - level) * 0.35;
  const orb = document.querySelector(`.voice[data-voice="${u.voice}"] .orb`);
  if (orb) orb.style.setProperty("--level", level.toFixed(3));
  if (u.playing && u.done && pos >= u.samples - 1) { u.playing = false; u.pausedAt = 0; ui.play.innerHTML = ICON.play; }
}

let looping = false;
function loop() {
  if (looping) return;
  looping = true;
  const tick = () => {
    drawMain();
    drawPlayingClip();
    const busy = (utt && (!utt.done || utt.playing || Math.abs(utt.scale - utt.samples) > 1 || level > 0.01)) || (clip.audio && !clip.audio.paused);
    if (busy) requestAnimationFrame(tick); else looping = false;
  };
  requestAnimationFrame(tick);
}

// ---------- demo events ----------
for (const b of document.querySelectorAll(".voice")) b.addEventListener("click", () => selectVoice(b.dataset.voice));
ui.speak.addEventListener("click", speak);
ui.text.addEventListener("input", updateCount);
ui.text.addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); speak(); } });
ui.speed.addEventListener("input", () => { ui.speedVal.textContent = `${(+ui.speed.value).toFixed(2)}×`; });
ui.play.addEventListener("click", () => {
  if (!utt) return;
  if (utt.playing) pauseMain();
  else { audioCtx(); playFrom(utt.pausedAt >= utt.samples - 1 ? 0 : utt.pausedAt); }
});
ui.wave.parentElement.addEventListener("click", e => {
  if (!utt || !utt.samples) return;
  const r = ui.wave.getBoundingClientRect();
  const pos = ((e.clientX - r.left) / r.width) * utt.scale;
  if (pos < utt.samples) { audioCtx(); playFrom(pos); }
});
ui.download.addEventListener("click", () => {
  if (!utt || !utt.samples) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(wavBlob(fullAudio(utt)));
  a.download = `teacher-in-the-loop-${utt.voice}.wav`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
});
$("#try").addEventListener("pointerenter", () => prefetch(voice), { once: true });
ui.text.addEventListener("focus", () => prefetch(voice), { once: true });
window.addEventListener("resize", () => { drawMain(); drawAllClips(); });

function wavBlob(f32) {
  const buf = new ArrayBuffer(44 + f32.length * 2), v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + f32.length * 2, true); str(8, "WAVEfmt "); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, SR, true); v.setUint32(28, SR * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, "data"); v.setUint32(40, f32.length * 2, true);
  for (let i = 0; i < f32.length; i++) v.setInt16(44 + 2 * i, Math.round(Math.max(-1, Math.min(1, f32[i])) * 32767), true);
  return new Blob([buf], { type: "audio/wav" });
}

ui.text.maxLength = MAX_CHARS;
ui.text.value = DEFAULT_TEXT[voice];
updateCount();
setExamples();
setSpeakButton("idle");
ui.play.innerHTML = ICON.play;

// ---------- samples: tabs and clip players ----------
const clip = { audio: null, el: null };

function peaksOf(el) {
  el._peaks ??= Uint8Array.from(atob(el.dataset.peaks), c => c.charCodeAt(0));
  return el._peaks;
}
function drawClip(el, played = 0) {
  const cv = el.querySelector("canvas");
  if (!cv.clientWidth) return;
  const [g, w, h] = sizeCanvas(cv);
  const p = peaksOf(el);
  bars(g, w, h, p.length, i => p[i] / 255, played);
}
function drawAllClips() {
  for (const el of document.querySelectorAll(".panel:not([hidden]) .clip")) drawClip(el, el === clip.el && clip.audio ? clip.audio.currentTime / el.dataset.dur : 0);
}
function drawPlayingClip() {
  if (clip.el && clip.audio) drawClip(clip.el, clip.audio.currentTime / (clip.audio.duration || +clip.el.dataset.dur));
}
function setClipIcon(el, playing) {
  el.classList.toggle("playing", playing);
  el.querySelector(".pb").innerHTML = playing ? ICON.pause : ICON.play;
}
function pauseClip() {
  if (clip.audio && !clip.audio.paused) { clip.audio.pause(); setClipIcon(clip.el, false); }
}
function playClip(el) {
  if (clip.el === el && clip.audio) {
    if (clip.audio.paused) { pauseMain(); clip.audio.play(); setClipIcon(el, true); loop(); } else pauseClip();
    return;
  }
  // switching system within the same sentence keeps the position, for A/B listening
  let t = 0;
  if (clip.el && clip.audio && !clip.audio.paused && clip.el.closest(".sent") === el.closest(".sent")) t = clip.audio.currentTime;
  if (clip.el) { pauseClip(); drawClip(clip.el, 0); }
  pauseMain();
  const a = new Audio(el.dataset.src);
  a.preload = "auto";
  if (t > 0) a.addEventListener("loadedmetadata", () => { a.currentTime = Math.min(t, a.duration - 0.05); }, { once: true });
  a.addEventListener("ended", () => { setClipIcon(el, false); drawClip(el, 0); });
  clip.audio = a;
  clip.el = el;
  a.play();
  setClipIcon(el, true);
  loop();
}
for (const el of document.querySelectorAll(".clip")) {
  el.querySelector(".pb").innerHTML = ICON.play;
  el.addEventListener("click", () => playClip(el));
}
function selectTab(id) {
  for (const t of document.querySelectorAll(".tab")) t.setAttribute("aria-selected", String(t.dataset.panel === id));
  for (const p of document.querySelectorAll(".panel")) p.hidden = p.id !== id;
  drawAllClips();
}
for (const t of document.querySelectorAll(".tab")) t.addEventListener("click", () => selectTab(t.dataset.panel));
drawAllClips();

// links like #listen or #if_sara: the Space page passes the fragment to this frame, but the browser does not
// always scroll there on load; a voice's id also opens its tab
function followHash() {
  const id = decodeURIComponent(location.hash.slice(1));
  if (!id) return;
  let target = document.getElementById(id);
  if (document.getElementById(`panel-${id}`)) { selectTab(`panel-${id}`); target = document.getElementById("listen"); }
  if (!target) return;
  // scroll this frame only (scrollIntoView may be ignored inside the Space's cross-origin frame), and again
  // once the Space page has sized the frame
  const go = () => window.scrollTo(0, target.getBoundingClientRect().top + window.scrollY);
  requestAnimationFrame(go);
  setTimeout(go, 400);
}
window.addEventListener("hashchange", followHash);
if (document.readyState === "complete") followHash(); else window.addEventListener("load", followHash, { once: true });
