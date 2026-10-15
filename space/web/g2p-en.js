// English text -> Kokoro phonemes: a port of Misaki 0.9.4's en.G2P (American, trf=False, eSpeak NG fallback,
// unk="") for the browser. spaCy is not available here: tokens are split by rules close to spaCy's English
// tokenizer and part-of-speech tags are guessed from the neighbouring words, so a few words that depend on the
// tag (noun/verb homographs, the direction of straight quotes) can differ from the Python pipeline. It matches
// that pipeline on 1,898 of 2,000 test sentences (99.7% of words).

const DIPHTHONGS = new Set("AIOQWYʤʧ");
const SUBTOKEN_RE = /^['‘’]+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['‘’]{2,}|\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:['‘’]\p{L})*|[^-_\p{L}'‘’\d]|['‘’]+$/gu;
const SUBTOKEN_JUNKS = new Set("',-._‘’/");
const PUNCTS = new Set(';:,.!?—…"“”');
const NON_QUOTE_PUNCTS = new Set(';:,.!?—…');
const PUNCT_TAGS = new Set([".", ",", "-LRB-", "-RRB-", "``", '""', "''", ":", "$", "#", "NFP"]);
const PUNCT_TAG_PHONEMES = { "-LRB-": "(", "-RRB-": ")", "``": "“", '""': "”", "''": "”" };
const CONSONANTS = new Set("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ");
const US_TAUS = new Set("AIOWYiuæɑəɛɪɹʊʌ");
const CURRENCIES = { "$": ["dollar", "cent"], "£": ["pound", "pence"], "€": ["euro", "cent"] };
const ORDINALS = new Set(["st", "nd", "rd", "th"]);
const ADD_SYMBOLS = { ".": "dot", "/": "slash" };
const SYMBOLS = { "%": "percent", "&": "and", "+": "plus", "@": "at" };
const PRIMARY = "ˈ", SECONDARY = "ˌ", STRESSES = "ˌˈ";
const VOWELS = new Set("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ");

const isAlpha = s => s.length > 0 && /^\p{L}+$/u.test(s);
const isDigitStr = s => /^[0-9]+$/.test(s);
const lower = s => s.toLowerCase();
const upper = s => s.toUpperCase();
const capitalize = s => s ? s[0].toUpperCase() + s.slice(1).toLowerCase() : s;
const lexOrd = c => c === "'" || c === "-" || (c >= "A" && c <= "Z") || (c >= "a" && c <= "z");
const isUpperCaseWord = w => w === upper(w);

function stressWeight(ps) { let n = 0; for (const c of ps || "") n += DIPHTHONGS.has(c) ? 2 : 1; return n; }

function restress(ps) {
  const ips = [...ps].map((p, i) => [i, p]);
  const moves = new Map();
  ips.forEach(([i, p]) => {
    if (STRESSES.includes(p)) {
      const j = ips.slice(i).find(([, v]) => VOWELS.has(v));
      if (j) moves.set(i, j[0]);
    }
  });
  for (const [i, j] of moves) ips[i] = [j - 0.5, ips[i][1]];
  return ips.sort((a, b) => a[0] - b[0]).map(x => x[1]).join("");
}

export function applyStress(ps, stress) {
  if (stress === null || stress === undefined || ps === null || ps === undefined) return ps;
  const has = s => ps.includes(s);
  const anyVowel = [...ps].some(c => VOWELS.has(c));
  if (stress < -1) return ps.replaceAll(PRIMARY, "").replaceAll(SECONDARY, "");
  if (stress === -1 || ((stress === 0 || stress === -0.5) && has(PRIMARY)))
    return ps.replaceAll(SECONDARY, "").replaceAll(PRIMARY, SECONDARY);
  if ((stress === 0 || stress === 0.5 || stress === 1) && !has(PRIMARY) && !has(SECONDARY))
    return anyVowel ? restress(SECONDARY + ps) : ps;
  if (stress >= 1 && !has(PRIMARY) && has(SECONDARY)) return ps.replaceAll(SECONDARY, PRIMARY);
  if (stress > 1 && !has(PRIMARY) && !has(SECONDARY)) return anyVowel ? restress(PRIMARY + ps) : ps;
  return ps;
}

// ---- num2words (English) for the forms Misaki uses: cardinal, ordinal, year, float ----
const ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
  "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const SCALES = [[1e18, "quintillion"], [1e15, "quadrillion"], [1e12, "trillion"], [1e9, "billion"], [1e6, "million"], [1e3, "thousand"]];
function under100(n) { return n < 20 ? ONES[n] : TENS[Math.floor(n / 10)] + (n % 10 ? "-" + ONES[n % 10] : ""); }
function under1000(n) {
  const h = Math.floor(n / 100), r = n % 100;
  if (!h) return under100(r);
  return ONES[h] + " hundred" + (r ? " and " + under100(r) : "");
}
export function cardinal(n) {
  n = Math.trunc(n);
  if (n < 0) return "minus " + cardinal(-n);
  if (n < 1000) return under1000(n);
  const parts = [];
  let rest = n;
  for (const [v, name] of SCALES) {
    if (rest >= v) { const q = Math.floor(rest / v); parts.push(cardinal(q) + " " + name); rest -= q * v; }
  }
  if (rest) parts.push((rest < 100 ? "and " : "") + under1000(rest));
  return parts.join(", ").replace(", and ", " and ");
}
const ORD = { one: "first", two: "second", three: "third", five: "fifth", eight: "eighth", nine: "ninth", twelve: "twelfth" };
export function ordinal(n) {
  const words = cardinal(n);
  const m = words.match(/([a-z]+)$/);
  const last = m[1];
  let o = ORD[last] || (last.endsWith("y") ? last.slice(0, -1) + "ieth" : last + "th");
  return words.slice(0, words.length - last.length) + o;
}
export function year(n) {
  const high = Math.floor(n / 100), low = n % 100;
  if (high === 0 || (high % 10 === 0 && low < 10) || high >= 100) return cardinal(n);
  const lowtext = low === 0 ? "hundred" : low < 10 ? "oh-" + cardinal(low) : cardinal(low);
  return cardinal(high) + " " + lowtext;
}
function floatWords(s) {
  const [a, b = ""] = s.split(".");
  let out = cardinal(parseInt(a || "0", 10));
  if (b) out += " point " + [...b].map(d => ONES[+d]).join(" ");
  return out;
}

// ---- Lexicon ----
function growDictionary(d) {
  const e = {};
  for (const [k, v] of Object.entries(d)) {
    if ([...k].length < 2) continue;
    if (k === lower(k)) { if (k !== capitalize(k)) e[capitalize(k)] = v; }
    else if (k === capitalize(lower(k))) e[lower(k)] = v;
  }
  return Object.assign(e, d);
}

function parentTag(tag) {
  if (tag === null || tag === undefined) return tag;
  if (tag.startsWith("VB")) return "VERB";
  if (tag.startsWith("NN")) return "NOUN";
  if (tag.startsWith("ADV") || tag.startsWith("RB")) return "ADV";
  if (tag.startsWith("ADJ") || tag.startsWith("JJ")) return "ADJ";
  return tag;
}

class Lexicon {
  constructor(gold, silver) {
    this.capStresses = [0.5, 2];
    this.golds = growDictionary(gold);
    this.silvers = growDictionary(silver);
  }
  has(d, w) { return Object.prototype.hasOwnProperty.call(d, w); }
  getNNP(word) {
    const ps = [];
    for (const c of word) {
      if (!/\p{L}/u.test(c)) continue;
      const p = this.has(this.golds, upper(c)) ? this.golds[upper(c)] : null;
      if (p === null || typeof p !== "string") return [null, null];
      ps.push(p);
    }
    let s = applyStress(ps.join(""), 0);
    const k = s.lastIndexOf(SECONDARY);
    if (k >= 0) s = s.slice(0, k) + PRIMARY + s.slice(k + 1);
    return [s, 3];
  }
  getSpecialCase(word, tag, stress, ctx) {
    if (tag === "ADD" && word in ADD_SYMBOLS) return this.lookup(ADD_SYMBOLS[word], null, -0.5, ctx);
    if (word in SYMBOLS) return this.lookup(SYMBOLS[word], null, null, ctx);
    const inner = word.replace(/^\.+|\.+$/g, "");
    if (inner.includes(".") && isAlpha(word.replaceAll(".", "")) && Math.max(...word.split(".").map(x => x.length)) < 3)
      return this.getNNP(word);
    if (word === "a" || word === "A") return [tag === "DT" ? "ɐ" : "ˈA", 4];
    if (["am", "Am", "AM"].includes(word)) {
      if (tag.startsWith("NN")) return this.getNNP(word);
      if (ctx.futureVowel === null || word !== "am" || (stress && stress > 0)) return [this.golds["am"], 4];
      return ["ɐm", 4];
    }
    if (["an", "An", "AN"].includes(word)) {
      if (word === "AN" && tag.startsWith("NN")) return this.getNNP(word);
      return ["ɐn", 4];
    }
    if (word === "I" && tag === "PRP") return [SECONDARY + "I", 4];
    if (["by", "By", "BY"].includes(word) && parentTag(tag) === "ADV") return ["bˈI", 4];
    if (word === "to" || word === "To" || (word === "TO" && (tag === "TO" || tag === "IN"))) {
      const v = ctx.futureVowel;
      return [v === null ? this.golds["to"] : v === false ? "tə" : "tʊ", 4];
    }
    if (word === "in" || word === "In" || (word === "IN" && tag !== "NNP")) {
      const s = ctx.futureVowel === null || tag !== "IN" ? PRIMARY : "";
      return [s + "ɪn", 4];
    }
    if (word === "the" || word === "The" || (word === "THE" && tag === "DT")) return [ctx.futureVowel === true ? "ði" : "ðə", 4];
    if (tag === "IN" && /^vs\.?$/i.test(word)) return this.lookup("versus", null, null, ctx);
    if (["used", "Used", "USED"].includes(word)) {
      if ((tag === "VBD" || tag === "JJ") && ctx.futureTo) return [this.golds["used"]["VBD"], 4];
      return [this.golds["used"]["DEFAULT"], 4];
    }
    return [null, null];
  }
  isKnown(word, tag) {
    if (this.has(this.golds, word) || word in SYMBOLS || this.has(this.silvers, word)) return true;
    if (!isAlpha(word) || ![...word].every(lexOrd)) return false;
    if ([...word].length === 1) return true;
    if (word === upper(word) && this.has(this.golds, lower(word))) return true;
    return word.slice(1) === upper(word.slice(1));
  }
  lookup(word, tag, stress, ctx) {
    let isNNP = null;
    if (word === upper(word) && !this.has(this.golds, word)) { word = lower(word); isNNP = tag === "NNP"; }
    let ps = this.has(this.golds, word) ? this.golds[word] : null, rating = 4;
    if (ps === null && !isNNP) { ps = this.has(this.silvers, word) ? this.silvers[word] : null; rating = 3; }
    if (ps !== null && typeof ps === "object") {
      if (ctx && ctx.futureVowel === null && "None" in ps) tag = "None";
      else if (!(tag in ps)) tag = parentTag(tag);
      ps = tag in ps ? ps[tag] : ps["DEFAULT"];
    }
    if (ps === null || ps === undefined || (isNNP && !ps.includes(PRIMARY))) {
      const [p, r] = this.getNNP(word);
      if (p !== null) return [p, r];
    }
    return [applyStress(ps === undefined ? null : ps, stress), rating];
  }
  _s(stem) {
    if (!stem) return null;
    const c = stem[stem.length - 1];
    if ("ptkfθ".includes(c)) return stem + "s";
    if ("szʃʒʧʤ".includes(c)) return stem + "ᵻz";
    return stem + "z";
  }
  stemS(word, tag, stress, ctx) {
    if (word.length < 3 || !word.endsWith("s")) return [null, null];
    let stem;
    if (!word.endsWith("ss") && this.isKnown(word.slice(0, -1), tag)) stem = word.slice(0, -1);
    else if ((word.endsWith("'s") || (word.length > 4 && word.endsWith("es") && !word.endsWith("ies"))) && this.isKnown(word.slice(0, -2), tag)) stem = word.slice(0, -2);
    else if (word.length > 4 && word.endsWith("ies") && this.isKnown(word.slice(0, -3) + "y", tag)) stem = word.slice(0, -3) + "y";
    else return [null, null];
    const [ps, rating] = this.lookup(stem, tag, stress, ctx);
    return [this._s(ps), rating];
  }
  _ed(stem) {
    if (!stem) return null;
    const c = stem[stem.length - 1];
    if ("pkfθʃsʧ".includes(c)) return stem + "t";
    if (c === "d") return stem + "ᵻd";
    if (c !== "t") return stem + "d";
    if (stem.length < 2) return stem + "ɪd";
    if (US_TAUS.has(stem[stem.length - 2])) return stem.slice(0, -1) + "ɾᵻd";
    return stem + "ᵻd";
  }
  stemEd(word, tag, stress, ctx) {
    if (word.length < 4 || !word.endsWith("d")) return [null, null];
    let stem;
    if (!word.endsWith("dd") && this.isKnown(word.slice(0, -1), tag)) stem = word.slice(0, -1);
    else if (word.length > 4 && word.endsWith("ed") && !word.endsWith("eed") && this.isKnown(word.slice(0, -2), tag)) stem = word.slice(0, -2);
    else return [null, null];
    const [ps, rating] = this.lookup(stem, tag, stress, ctx);
    return [this._ed(ps), rating];
  }
  _ing(stem) {
    if (!stem) return null;
    if (stem.length > 1 && stem[stem.length - 1] === "t" && US_TAUS.has(stem[stem.length - 2])) return stem.slice(0, -1) + "ɾɪŋ";
    return stem + "ɪŋ";
  }
  stemIng(word, tag, stress, ctx) {
    if (word.length < 5 || !word.endsWith("ing")) return [null, null];
    let stem;
    if (word.length > 5 && this.isKnown(word.slice(0, -3), tag)) stem = word.slice(0, -3);
    else if (this.isKnown(word.slice(0, -3) + "e", tag)) stem = word.slice(0, -3) + "e";
    else if (word.length > 5 && /([bcdgklmnprstvxz])\1ing$|cking$/.test(word) && this.isKnown(word.slice(0, -4), tag)) stem = word.slice(0, -4);
    else return [null, null];
    const [ps, rating] = this.lookup(stem, tag, stress, ctx);
    return [this._ing(ps), rating];
  }
  getWord(word, tag, stress, ctx) {
    let [ps, rating] = this.getSpecialCase(word, tag, stress, ctx);
    if (ps !== null) return [ps, rating];
    const wl = lower(word);
    if (word.length > 1 && isAlpha(word.replaceAll("'", "")) && word !== lower(word) && (tag !== "NNP" || word.length > 7)
      && !this.has(this.golds, word) && !this.has(this.silvers, word)
      && (word === upper(word) || word.slice(1) === lower(word.slice(1)))
      && (this.has(this.golds, wl) || this.has(this.silvers, wl)
        || [this.stemS, this.stemEd, this.stemIng].some(fn => fn.call(this, wl, tag, stress, ctx)[0])))
      word = wl;
    if (this.isKnown(word, tag)) return this.lookup(word, tag, stress, ctx);
    if (word.endsWith("s'") && this.isKnown(word.slice(0, -2) + "'s", tag)) return this.lookup(word.slice(0, -2) + "'s", tag, stress, ctx);
    if (word.endsWith("'") && this.isKnown(word.slice(0, -1), tag)) return this.lookup(word.slice(0, -1), tag, stress, ctx);
    let r = this.stemS(word, tag, stress, ctx); if (r[0] !== null) return r;
    r = this.stemEd(word, tag, stress, ctx); if (r[0] !== null) return r;
    r = this.stemIng(word, tag, stress === null ? 0.5 : stress, ctx); if (r[0] !== null) return r;
    return [null, null];
  }
  static isCurrency(word) {
    if (!word.includes(".")) return true;
    if (word.split(".").length - 1 > 1) return false;
    const cents = word.split(".")[1];
    return cents.length < 3;
  }
  getNumber(word, currency, isHead, numFlags) {
    const sm = word.match(/[a-z']+$/);
    const suffix = sm ? sm[0] : null;
    if (suffix) word = word.slice(0, -suffix.length);
    const result = [];
    if (word.startsWith("-")) { result.push(this.lookup("minus", null, null, null)); word = word.slice(1); }
    const extendNum = (num, first = true, escape = false) => {
      const splits = (escape ? num : cardinal(parseInt(num, 10))).split(/[^a-z]+/);
      splits.forEach((w, i) => {
        if (w !== "and" || numFlags.includes("&")) {
          if (first && i === 0 && splits.length > 1 && w === "one" && numFlags.includes("a")) result.push(["ə", 4]);
          else result.push(this.lookup(w, null, w === "point" ? -2 : null, null));
        } else if (w === "and" && numFlags.includes("n") && result.length) {
          result[result.length - 1] = [result[result.length - 1][0] + "ən", result[result.length - 1][1]];
        }
      });
    };
    if (isDigitStr(word) && ORDINALS.has(suffix)) extendNum(ordinal(parseInt(word, 10)), true, true);
    else if (!result.length && word.length === 4 && !(currency in CURRENCIES) && isDigitStr(word)) extendNum(year(parseInt(word, 10)), true, true);
    else if (!isHead && !word.includes(".")) {
      const num = word.replaceAll(",", "");
      if (num[0] === "0" || num.length > 3) [...num].forEach(n => extendNum(n, false));
      else if (num.length === 3 && !num.endsWith("00")) {
        extendNum(num[0]);
        if (num[1] === "0") { result.push(this.lookup("O", null, -2, null)); extendNum(num[2], false); }
        else extendNum(num.slice(1), false);
      } else extendNum(num);
    } else if (word.split(".").length - 1 > 1 || !isHead) {
      let first = true;
      for (const num of word.replaceAll(",", "").split(".")) {
        if (!num) { /* pass */ }
        else if (num[0] === "0" || (num.length !== 2 && [...num.slice(1)].some(n => n !== "0"))) [...num].forEach(n => extendNum(n, false));
        else extendNum(num, first);
        first = false;
      }
    } else if (currency in CURRENCIES && Lexicon.isCurrency(word)) {
      let pairs = word.replaceAll(",", "").split(".").map((num, i) => [num ? parseInt(num, 10) : 0, CURRENCIES[currency][i]]).filter(p => p[1] !== undefined);
      if (pairs.length > 1) { if (pairs[1][0] === 0) pairs = pairs.slice(0, 1); else if (pairs[0][0] === 0) pairs = pairs.slice(1); }
      pairs.forEach(([num, unit], i) => {
        if (i > 0) result.push(this.lookup("and", null, null, null));
        extendNum(String(num), i === 0);
        result.push(Math.abs(num) !== 1 && unit !== "pence" ? this.stemS(unit + "s", null, null, null) : this.lookup(unit, null, null, null));
      });
    } else {
      let w;
      if (isDigitStr(word)) w = cardinal(parseInt(word, 10));
      else if (!word.includes(".")) w = ORDINALS.has(suffix) ? ordinal(parseInt(word.replaceAll(",", ""), 10)) : cardinal(parseInt(word.replaceAll(",", ""), 10));
      else {
        const x = word.replaceAll(",", "");
        w = x[0] === "." ? "point " + [...x.slice(1)].map(n => cardinal(+n)).join(" ") : floatWords(x);
      }
      extendNum(w, true, true);
    }
    if (!result.length) return [null, null];
    if (result.some(r => r[0] === null)) return [null, null];
    let ps = result.map(r => r[0]).join(" ");
    const rating = Math.min(...result.map(r => r[1]));
    if (suffix === "s" || suffix === "'s") return [this._s(ps), rating];
    if (suffix === "ed" || suffix === "'d") return [this._ed(ps), rating];
    if (suffix === "ing") return [this._ing(ps), rating];
    return [ps, rating];
  }
  appendCurrency(ps, currency) {
    if (!currency) return ps;
    const c = CURRENCIES[currency];
    const s = c ? this.stemS(c[0] + "s", null, null, null)[0] : null;
    return s ? `${ps} ${s}` : ps;
  }
  static isNumber(word, isHead) {
    if (![...word].some(c => /[0-9]/.test(c))) return false;
    for (const s of ["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"]) {
      if (word.endsWith(s)) { word = word.slice(0, -s.length); break; }
    }
    return [...word].every((c, i) => /[0-9]/.test(c) || ",.".includes(c) || (isHead && i === 0 && c === "-"));
  }
  call(tk, ctx) {
    let word = (tk.alias === null ? tk.text : tk.alias).replaceAll("‘", "'").replaceAll("’", "'");
    word = word.normalize("NFKC");
    const stress = word === lower(word) ? null : this.capStresses[word === upper(word) ? 1 : 0];
    let [ps, rating] = this.getWord(word, tk.tag, stress, ctx);
    if (ps !== null) return [applyStress(this.appendCurrency(ps, tk.currency), tk.stress), rating];
    if (Lexicon.isNumber(word, tk.isHead)) {
      [ps, rating] = this.getNumber(word, tk.currency, tk.isHead, tk.numFlags);
      return [applyStress(ps, tk.stress), rating];
    }
    return [null, null];
  }
}

// ---- eSpeak NG fallback (Misaki's EspeakFallback, American) ----
const E2M = Object.entries({
  "ʔˌn̩": "ʔn", "ʔn̩": "ʔn", "a^ɪ": "I", "a^ʊ": "W", "d^ʒ": "ʤ", "e^ɪ": "A", "e": "A", "t^ʃ": "ʧ",
  "ɔ^ɪ": "Y", "ə^l": "ᵊl", "ʲo": "jo", "ʲə": "jə", "ʲ": "", "ɚ": "əɹ", "r": "ɹ", "x": "k", "ç": "k", "ɐ": "ə",
  "ɬ": "l", "̃": "",
}).sort((a, b) => [...b[0]].length - [...a[0]].length);

export function makeEspeakChunk(m, worker, voice) {
  const dec = new TextDecoder();
  const MODIFIERS = new Set(["ː", "ˑ", "̪", "̃", "ʲ", "ʰ", "ʷ", "̩", "̯"]);
  function cstr(ptr) { const h = m.HEAPU8; let e = ptr; while (h[e]) e++; return dec.decode(h.subarray(ptr, e)); }
  return function chunk(text, removeFlags) {
    worker.set_voice(voice);
    const p = worker.text_to_phonemes(text, 1);
    let line = cstr(m.getPointer(p)).replace(/ \| /g, " ");
    line = line.trim().replace(/\n/g, " ").replace(/  /g, " ").replace(/_+/g, "_").replace(/_ /g, " ");
    if (removeFlags) line = line.replace(/\([a-z-]+\)/g, "");
    const words = [];
    for (let word of line.split(" ")) {
      word = word.trim();
      if (!word) continue;
      words.push(word.split("_").map(ph => {
        const cs = [...ph];
        let out = "";
        for (let i = 0; i < cs.length; i++) {
          out += cs[i];
          const nx = cs[i + 1];
          if (nx !== undefined && cs[i] !== "ˈ" && cs[i] !== "ˌ" && nx !== "ˈ" && nx !== "ˌ" && !MODIFIERS.has(nx)) out += "^";
        }
        return out;
      }).join(""));
    }
    return words.join(" ");
  };
}

const MARKS = ';:,.!?¡¿—…"«»“”(){}[]';
const esc = s => s.replace(/[-\/\\^$*+?.()|[\]{}]/g, "\\$&");
const MARKS_RE = new RegExp(`(\\s*[${esc(MARKS)}]+\\s*)+`, "g");

// phonemizer's EspeakBackend.phonemize with preserve_punctuation=True, for one line of text
export function phonemizeLine(text, chunk, removeFlags) {
  const matches = [...text.matchAll(MARKS_RE)];
  if (!matches.length) return chunk(text, removeFlags);
  if (matches.length === 1 && matches[0][0] === text) return text;
  const marks = matches.map(mt => mt[0]);
  const parts = [];
  let line = text;
  for (const mk of marks) { const k = line.indexOf(mk); parts.push(line.slice(0, k)); line = line.slice(k + mk.length); }
  parts.push(line);
  let out = "";
  parts.forEach((part, i) => { if (part.trim()) out += chunk(part, removeFlags); if (i < marks.length) out += marks[i]; });
  return out;
}

function espeakFallback(text, chunk) {
  let ps = phonemizeLine(text, chunk, false).trim();
  if (!ps) return [null, null];
  for (const [a, b] of E2M) ps = ps.split(a).join(b);
  ps = ps.replace(/(\S)̩/gu, "ᵊ$1").replaceAll("̩", "");
  ps = ps.replaceAll("o^ʊ", "O").replaceAll("ɜːɹ", "ɜɹ").replaceAll("ɜː", "ɜɹ").replaceAll("ɪə", "iə").replaceAll("ː", "");
  ps = ps.replaceAll("o", "ɔ");
  ps = ps.replaceAll("ɾ", "T").replaceAll("ʔ", "t");
  return [ps.replaceAll("^", ""), 2];
}

// ---- tokenization (rules close to spaCy's English tokenizer) and tag guesses ----
const PREFIX_CHARS = new Set(['"', "'", "(", "[", "{", "“", "‘", "$", "£", "€", "#", "¿", "¡", "«"]);
const SUFFIX_CHARS = new Set([",", ";", ":", "!", "?", ")", "]", "}", '"', "”", "’", "'", "%", "»", "…"]);
// spaCy's English exceptions that keep their final period
const ABBREV = new Set(["a.m.", "p.m.", "e.g.", "E.g.", "E.G.", "i.e.", "I.e.", "I.E.", "vs.", "v.s.", "co.", "Co.",
  "Adm.", "Bros.", "Corp.", "D.C.", "Dr.", "Gen.", "Gov.", "Inc.", "Jr.", "Ltd.", "Md.", "Messrs.", "Mo.", "Mont.",
  "Mr.", "Mrs.", "Ms.", "Ph.D.", "Prof.", "Rep.", "Rev.", "Sen.", "St.", "Mt.", "Jan.", "Feb.", "Mar.", "Apr.",
  "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "Calif.", "Conn.", "Fla.", "Ill.", "Mass.",
  "Mich.", "Minn.", "Tenn.", "Wash.", "Wis."]);
const SPLIT_CONTRACTIONS = /^(.+?)(n't|'s|'re|'ve|'ll|'d|'m|’s|’re|’ve|’ll|’d|’m|n’t)$/i;
const DROPPED_G = /^(?:goin|doin|nothin|somethin|havin|lovin|comin|talkin|gettin|makin|sayin|tryin|runnin)'$/i;

// spaCy splits a final period after a lowercase letter, a digit, %, -, +, a quote or punctuation, or two capitals
function splitsPeriod(w) {
  if (ABBREV.has(w) || /^\p{Ll}\.$/u.test(w)) return false;
  const b = w.slice(0, -1);
  return /[\p{Ll}0-9%²\-+'"”’)\]!?,;:.]$/u.test(b) || /\p{Lu}\p{Lu}$/u.test(b);
}

function splitWord(w) {
  const pre = [], post = [];
  while (w.length > 1 && PREFIX_CHARS.has(w[0]) && !(w[0] === "'" && /^'(?:\d0s|[smdt]|ll|ve|re)$/i.test(w))) { pre.push(w[0]); w = w.slice(1); }
  let changed = true;
  while (changed && w.length > 1) {
    changed = false;
    if (w.endsWith("...")) { post.unshift("..."); w = w.slice(0, -3); changed = true; continue; }
    const c = w[w.length - 1];
    if (c === "." ? splitsPeriod(w) : SUFFIX_CHARS.has(c) && !(c === "'" && DROPPED_G.test(w))) {
      post.unshift(c); w = w.slice(0, -1); changed = true;
    }
  }
  const mid = [];
  const cm = w.match(SPLIT_CONTRACTIONS);
  if (cm && /\p{L}$/u.test(cm[1]) && !ABBREV.has(w)) mid.push(cm[1], cm[2]); else if (w) mid.push(w);
  // infix: hyphen between letters, as spaCy does
  const out = [];
  for (const t of mid) out.push(...t.split(/(?<=\p{L})(-)(?=\p{L})|(—|–)/u).filter(x => x));
  return [...pre, ...out, ...post];
}

const DETS = new Set(["a", "an", "the", "this", "these", "those", "every", "some", "no", "any", "each", "his", "her",
  "its", "their", "our", "my", "your", "another", "many", "several", "such", "whose"]);
const PREPS = new Set(["in", "on", "at", "of", "for", "with", "from", "by", "into", "during", "after", "before",
  "about", "against", "between", "through", "under", "over", "since", "until", "upon", "within", "without", "as",
  "than", "because", "while", "if", "among", "across", "toward", "towards", "around", "despite", "via", "per"]);
const SUBJECTS = new Set(["i", "he", "she", "it", "we", "they", "you", "who", "which"]);
const OBJECTS = new Set(["him", "her", "them", "us", "me", "it", "this", "these", "those", "his", "their", "its", "our",
  "my", "your", "the", "a", "an"]);
const MODALS = new Set(["to", "will", "would", "can", "could", "should", "shall", "may", "might", "must", "do", "does",
  "did", "not", "n't", "never", "cannot", "'ll", "'d"]);
const NUMBER_WORDS = new Set(["hundred", "thousand", "million", "billion", "trillion"]);
const THAT_NOUNS = new Set(["year", "day", "time", "kind", "same", "one", "night", "week", "month", "season", "point",
  "way", "morning", "evening", "afternoon", "summer", "winter", "spring", "autumn", "period", "moment", "man", "woman"]);
const PARTICLES = new Set(["up", "down", "out", "off"]);
const PHRASAL = new Set(["fill", "come", "came", "get", "got", "go", "went", "put", "set", "give", "gave", "take", "took",
  "move", "check", "log", "sign", "turn", "hand", "break", "broke", "call", "drop", "fit", "join", "stay", "swear",
  "settle", "kick", "kicked", "pitch", "chip", "cave", "give", "let", "lock", "plug", "phase", "zoom", "tune"]);
const BE = new Set(["is", "was", "were", "are", "be", "been", "being", "'s", "'re"]);
const isWordTok = t => t && /\p{L}/u.test(t.text);

function guessTag(toks, i) {
  const text = toks[i].text, l = lower(text);
  const prev = toks[i - 1], next = toks[i + 1];
  const pl = prev ? lower(prev.text) : null, nl = next ? lower(next.text) : null;
  if (text === "(" || text === "[" || text === "{") return "-LRB-";
  if (text === ")" || text === "]" || text === "}") return "-RRB-";
  if (text === "“") return "``";
  if (text === "”" || text === "'" || text === "’") return "''";
  if (text === ",") return ",";
  if (text === "." || text === "!" || text === "?") return ".";
  if (text === ":" || text === ";" || text === "—" || text === "–" || text === "..." || text === "…") return ":";
  if (text === "-") return prev && prev.whitespace === "" ? "HYPH" : "NFP";
  if (text in CURRENCIES) return "$";
  if (text === "#") return "#";
  if (/\d/.test(text) && /^-?[\d,.]+$/.test(text)) return "CD";
  if (NUMBER_WORDS.has(l) && prev && /\d/.test(prev.text)) return "CD";
  if (/^(?:https?:\/\/|www\.)|^[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|edu|gov|io|co|uk|it|de|fr|info)$/i.test(text)) return "ADD";
  if ((l === "am" || l === "pm") && prev && /\d/.test(prev.text)) return "NN";
  if (l === "that") {
    if (!prev || !isWordTok(prev) || PREPS.has(pl) || pl === "to" || PARTICLES.has(pl) || !isWordTok(next) || THAT_NOUNS.has(nl)) return "DT";
    return "IN";
  }
  if (DETS.has(l)) return "DT";
  if (/^vs\.?$/.test(l)) return "IN";
  // "taken in by", "coming in for": a particle, stressed
  if (l === "in" && prev && (/(?:ed|ing|en)$/.test(pl) || PHRASAL.has(pl)) && (!next || !/[\p{L}\d]/u.test(next.text) || PREPS.has(nl) || nl === "to")) return "RP";
  if (l === "to") return "TO";
  if (PREPS.has(l)) return "IN";
  if (text === "I") return prev && /^\p{Lu}\p{Ll}/u.test(prev.text) && i > 1 && (!next || !isWordTok(next)) ? "NNP" : "PRP";
  if (/^\p{Lu}{2,}$/u.test(text)) return "NNP";
  if (SUBJECTS.has(l) || OBJECTS.has(l) && l.length < 5) return "PRP";
  if (/^\p{Lu}/u.test(text) && i > 0 && !(prev && /^[.!?“"]$/.test(prev.text))) return "NNP";
  // a noun or a verb, from the neighbours (only homographs in the lexicon depend on it)
  if (l === "used") return prev && BE.has(pl) ? "VBN" : "VBD";
  if (prev && /^\d/.test(prev.text)) return "NN";
  if (prev && (DETS.has(pl) || PREPS.has(pl) || pl === "'s" || pl === "’s"))
    return isWordTok(next) && !PREPS.has(nl) && nl !== "to" && !MODALS.has(nl) && !BE.has(nl) && !/^\p{Lu}{2}/u.test(next.text) && nl !== "and" ? "JJ" : "NN";
  if (nl === "of") return "NN";
  if (prev && (MODALS.has(pl) || SUBJECTS.has(pl) || (/ly$/.test(pl) && pl.length > 4))) return /s$/.test(l) ? "VBZ" : "VB";
  if (prev && /^\p{Lu}\p{Ll}/u.test(prev.text) && i > 1 && /s$/.test(l) && next && OBJECTS.has(nl)) return "VBZ";
  if (next && OBJECTS.has(nl) && !DETS.has(pl)) return /s$/.test(l) ? "VBZ" : "VB";
  if (/ed$/.test(l)) return "VBD";
  if (/ing$/.test(l)) return "VBG";
  return "X";
}

function tokenize(text) {
  const toks = [];
  const re = /(\S+)(\s*)/g;
  let mt;
  while ((mt = re.exec(text))) {
    const pieces = splitWord(mt[1]);
    pieces.forEach((p, j) => toks.push({ text: p, whitespace: j === pieces.length - 1 ? (mt[2] ? " " : "") : "" }));
  }
  // straight double quotes: alternate opening and closing
  let open = true;
  toks.forEach((t, i) => {
    t.tag = guessTag(toks, i);
    if (t.text === '"') { t.tag = open ? "``" : "''"; open = !open; }
  });
  return toks;
}

const mk = (o) => Object.assign({ text: "", tag: "X", whitespace: "", phonemes: null, alias: null, stress: null,
  currency: null, numFlags: "", prespace: false, rating: null, isHead: true }, o);

function mergeTokens(tokens, unk) {
  const stress = new Set(tokens.filter(t => t.stress !== null).map(t => t.stress));
  const currency = tokens.filter(t => t.currency !== null).map(t => t.currency);
  let phonemes = null;
  if (unk !== null && unk !== undefined) {
    phonemes = "";
    for (const tk of tokens) {
      if (tk.prespace && phonemes && !/\s$/.test(phonemes) && tk.phonemes) phonemes += " ";
      phonemes += tk.phonemes === null ? unk : tk.phonemes;
    }
  }
  const weight = t => [...t.text].reduce((s, c) => s + (c === lower(c) ? 1 : 2), 0);
  let best = tokens[0];
  for (const t of tokens) if (weight(t) > weight(best)) best = t;
  const ratings = tokens.map(t => t.rating);
  return mk({
    text: tokens.slice(0, -1).map(t => t.text + t.whitespace).join("") + tokens[tokens.length - 1].text,
    tag: best.tag, whitespace: tokens[tokens.length - 1].whitespace, phonemes,
    isHead: tokens[0].isHead, stress: stress.size === 1 ? [...stress][0] : null,
    currency: currency.length ? currency.sort().slice(-1)[0] : null,
    numFlags: [...new Set(tokens.flatMap(t => [...t.numFlags]))].sort().join(""),
    prespace: tokens[0].prespace, rating: ratings.includes(null) ? null : Math.min(...ratings),
  });
}

function retokenize(tokens) {
  const words = [];
  let currency = null;
  tokens.forEach((token, i) => {
    let tks;
    if (token.alias === null && token.phonemes === null) {
      const subs = token.text.match(SUBTOKEN_RE) || [];
      tks = subs.map(t => mk({ text: t, tag: token.tag, whitespace: "", isHead: true, numFlags: token.numFlags, stress: token.stress }));
    } else tks = [token];
    if (!tks.length) return;
    tks[tks.length - 1].whitespace = token.whitespace;
    tks.forEach((tk, j) => {
      if (tk.alias !== null || tk.phonemes !== null) { /* pass */ }
      else if (tk.tag === "$" && tk.text in CURRENCIES) { currency = tk.text; tk.phonemes = ""; tk.rating = 4; }
      else if (tk.tag === ":" && (tk.text === "-" || tk.text === "–")) { tk.phonemes = "—"; tk.rating = 3; }
      else if (PUNCT_TAGS.has(tk.tag) && ![...tk.text].every(c => { const o = lower(c).charCodeAt(0); return o >= 97 && o <= 122; })) {
        tk.phonemes = PUNCT_TAG_PHONEMES[tk.tag] ?? [...tk.text].filter(c => PUNCTS.has(c)).join("");
        tk.rating = 4;
      } else if (currency !== null) {
        if (tk.tag !== "CD") currency = null;
        else if (j + 1 === tks.length && (i + 1 === tokens.length || tokens[i + 1].tag !== "CD")) tk.currency = currency;
      } else if (j > 0 && j < tks.length - 1 && tk.text === "2" && /\p{L}/u.test(tks[j - 1].text.slice(-1) + tks[j + 1].text[0]) && /^\p{L}+$/u.test(tks[j - 1].text.slice(-1) + tks[j + 1].text[0])) {
        tk.alias = "to";
      }
      if (tk.alias !== null || tk.phonemes !== null) words.push(tk);
      else if (words.length && Array.isArray(words[words.length - 1]) && !words[words.length - 1].slice(-1)[0].whitespace) {
        tk.isHead = false; words[words.length - 1].push(tk);
      } else words.push(tk.whitespace ? tk : [tk]);
    });
  });
  return words.map(w => Array.isArray(w) && w.length === 1 ? w[0] : w);
}

function tokenContext(ctx, ps, token) {
  let vowel = ctx.futureVowel;
  if (ps) {
    for (const c of ps) {
      if (VOWELS.has(c) || CONSONANTS.has(c) || NON_QUOTE_PUNCTS.has(c)) { vowel = NON_QUOTE_PUNCTS.has(c) ? null : VOWELS.has(c); break; }
    }
  }
  const futureTo = token.text === "to" || token.text === "To" || (token.text === "TO" && (token.tag === "TO" || token.tag === "IN"));
  return { futureVowel: vowel, futureTo };
}

function resolveTokens(tokens) {
  const text = tokens.slice(0, -1).map(t => t.text + t.whitespace).join("") + tokens[tokens.length - 1].text;
  const classes = new Set([...text].filter(c => !SUBTOKEN_JUNKS.has(c)).map(c => /\p{L}/u.test(c) ? 0 : /[0-9]/.test(c) ? 1 : 2));
  const prespace = text.includes(" ") || text.includes("/") || classes.size > 1;
  tokens.forEach((tk, i) => {
    if (tk.phonemes === null) {
      if (i === tokens.length - 1 && NON_QUOTE_PUNCTS.has(tk.text)) { tk.phonemes = tk.text; tk.rating = 3; }
      else if ([...tk.text].every(c => SUBTOKEN_JUNKS.has(c))) { tk.phonemes = ""; tk.rating = 3; }
    } else if (i > 0) tk.prespace = prespace;
  });
  if (prespace) return;
  let indices = tokens.map((tk, i) => tk.phonemes ? [tk.phonemes.includes(PRIMARY) ? 1 : 0, stressWeight(tk.phonemes), i] : null).filter(x => x);
  if (indices.length === 2 && [...tokens[indices[0][2]].text].length === 1) {
    const i = indices[1][2];
    tokens[i].phonemes = applyStress(tokens[i].phonemes, -0.5);
    return;
  }
  if (indices.length < 2 || indices.reduce((s, x) => s + x[0], 0) <= Math.floor((indices.length + 1) / 2)) return;
  indices = indices.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]).slice(0, Math.floor(indices.length / 2));
  for (const [, , i] of indices) tokens[i].phonemes = applyStress(tokens[i].phonemes, -0.5);
}

export function makeEnglish(m, worker, gold, silver) {
  const lexicon = new Lexicon(gold, silver);
  const chunk = makeEspeakChunk(m, worker, "en-us");
  const fallback = tk => espeakFallback(tk.text, chunk);
  const unk = "";
  return function g2p(text) {
    text = text.replace(/^\s+/, "");
    let tokens = tokenize(text).map(t => mk(t));
    tokens = retokenize(tokens);
    let ctx = { futureVowel: null, futureTo: false };
    for (let i = tokens.length - 1; i >= 0; i--) {
      const w = tokens[i];
      if (!Array.isArray(w)) {
        if (w.phonemes === null) [w.phonemes, w.rating] = lexicon.call(w, ctx);
        if (w.phonemes === null) [w.phonemes, w.rating] = fallback(w);
        ctx = tokenContext(ctx, w.phonemes, w);
        continue;
      }
      let left = 0, right = w.length, shouldFallback = false;
      while (left < right) {
        let tk = null;
        if (!w.slice(left, right).some(t => t.alias !== null || t.phonemes !== null)) tk = mergeTokens(w.slice(left, right), null);
        const [ps, rating] = tk === null ? [null, null] : lexicon.call(tk, ctx);
        if (ps !== null) {
          w[left].phonemes = ps; w[left].rating = rating;
          for (const x of w.slice(left + 1, right)) { x.phonemes = ""; x.rating = rating; }
          ctx = tokenContext(ctx, ps, tk);
          right = left; left = 0;
        } else if (left + 1 < right) left += 1;
        else {
          right -= 1;
          const t = w[right];
          if (t.phonemes === null) {
            if ([...t.text].every(c => SUBTOKEN_JUNKS.has(c))) { t.phonemes = ""; t.rating = 3; }
            else { shouldFallback = true; break; }
          }
          left = 0;
        }
      }
      if (shouldFallback) {
        const tk = mergeTokens(w, null);
        [w[0].phonemes, w[0].rating] = fallback(tk);
        for (let j = 1; j < w.length; j++) { w[j].phonemes = ""; w[j].rating = w[0].rating; }
      } else resolveTokens(w);
    }
    tokens = tokens.map(t => Array.isArray(t) ? mergeTokens(t, unk) : t);
    for (const tk of tokens) if (tk.phonemes) tk.phonemes = tk.phonemes.replaceAll("ɾ", "T").replaceAll("ʔ", "t");
    return tokens.map(tk => (tk.phonemes === null ? unk : tk.phonemes) + tk.whitespace).join("").trim();
  };
}
