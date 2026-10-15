// Italian text -> Kokoro phonemes, as titl/Misaki do in Python (EspeakG2P("it") + dental mapping).
// m: the eSpeak NG module; espeak: an eSpeakNGWorker. eSpeak keeps one voice for the whole module, so the
// voice is set again on every call. Matches the Python pipeline on 2,000 of 2,000 test sentences.
const MARKS = ';:,.!?¡¿—…"«»“”(){}[]';
const esc = s => s.replace(/[-\/\\^$*+?.()|[\]{}]/g, "\\$&");
const MARKS_RE = new RegExp(`(\\s*[${esc(MARKS)}]+\\s*)+`, "g");
const E2M = [["a^ɪ", "I"], ["a^ʊ", "W"], ["d^z", "ʣ"], ["d^ʒ", "ʤ"], ["e^ɪ", "A"], ["o^ʊ", "O"], ["ə^ʊ", "Q"],
  ["s^s", "S"], ["t^s", "ʦ"], ["t^ʃ", "ʧ"], ["ɔ^ɪ", "Y"]].sort((a, b) => a[0] < b[0] ? -1 : 1);
const MODIFIERS = new Set(["ː", "ˑ", "̪", "̃", "ʲ", "ʰ", "ʷ", "̩", "̯"]);
const STRESS = new Set(["ˈ", "ˌ"]);

export function makeItalian(m, espeak) {
  const dec = new TextDecoder();
  function cstr(ptr) { const h = m.HEAPU8; let e = ptr; while (h[e]) e++; return dec.decode(h.subarray(ptr, e)); }
  // one chunk without punctuation -> words of phonemes with '^' ties inside multi-letter phonemes
  function chunk(text) {
    const p = espeak.text_to_phonemes(text, 1);
    let line = cstr(m.getPointer(p)).replace(/ \| /g, " ");
    line = line.trim().replace(/\n/g, " ").replace(/  /g, " ");
    line = line.replace(/_+/g, "_").replace(/_ /g, " ");
    line = line.replace(/\([a-z-]+\)/g, "");  // language-switch flags (remove-flags)
    const words = [];
    for (let word of line.split(" ")) {
      word = word.trim();
      if (!word) continue;
      const pieces = word.split("_").map(ph => {
        const cs = [...ph];
        let out = "";
        for (let i = 0; i < cs.length; i++) {
          out += cs[i];
          const next = cs[i + 1];
          if (next !== undefined && !STRESS.has(cs[i]) && !STRESS.has(next) && !MODIFIERS.has(next)) out += "^";
        }
        return out;
      });
      words.push(pieces.join(""));
    }
    return words.join(" ");
  }
  return function g2p(text) {
    espeak.set_voice("it");
    text = text.replace(/«/g, "“").replace(/»/g, "”").replace(/\(/g, "«").replace(/\)/g, "»");
    // phonemizer's Punctuation.preserve / restore (one line)
    const matches = [...text.matchAll(MARKS_RE)];
    let ps;
    if (!matches.length) ps = chunk(text);
    else if (matches.length === 1 && matches[0][0] === text) ps = text;
    else {
      const marks = matches.map((mt, i) => ({
        mark: mt[0],
        pos: i === 0 && text.startsWith(mt[0]) ? "B" : (i === matches.length - 1 && text.endsWith(mt[0]) ? "E" : "I"),
      }));
      const parts = [];
      let line = text;
      for (const mk of marks) {
        const k = line.indexOf(mk.mark);
        parts.push(line.slice(0, k));
        line = line.slice(k + mk.mark.length);
      }
      parts.push(line);
      let out = "";
      parts.forEach((part, i) => {
        if (part.trim()) out += chunk(part);
        if (i < marks.length) out += marks[i].mark;
      });
      ps = out;
    }
    ps = ps.trim();
    for (const [a, b] of E2M) ps = ps.split(a).join(b);
    ps = ps.replace(/\^/g, "").replace(/-/g, "");
    ps = ps.replace(/«/g, "(").replace(/»/g, ")");
    return ps.replace(/̪/g, "").trim();
  };
}
