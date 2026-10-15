"""Paired UTMOS comparison of two decoders on the same held-out rows and excitation seeds (teacher conditioning).

  python -m eval.pair VOICE A=DECODER.pt B=DECODER.pt [--n 200] [--out FILE]

Both decoders render the first N held-out rows as in eval.decoder (excitation seed 1000 + row). Reports B - A:
the mean per-row difference with its standard error, the paired t-test, the number of rows on which B scores
higher and the exact sign test (eval/stats.py), and writes the per-row scores to the JSON file.
"""
import argparse

import numpy as np

from training.common import VOICES, device
from training.utmos import Utmos

from .common import heldout, labelled, load_decoder, rounded, save_json
from .decoder import score
from .stats import paired


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("voice", choices=sorted(VOICES))
    ap.add_argument("a", metavar="A=DECODER.pt")
    ap.add_argument("b", metavar="B=DECODER.pt")
    ap.add_argument("--n", type=int, default=200, help="first N held-out rows")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="JSON file (default out/eval/pair-VOICE-A-B.json)")
    a = ap.parse_args(argv)
    dev = device(a.device)
    (la, pa), (lb, pb) = labelled([a.a, a.b])
    if la == lb:
        raise SystemExit("the two decoders need different labels")
    rows = heldout(a.voice, a.n)
    utmos = Utmos(dev)
    scores = {}
    for label, path in ((la, pa), (lb, pb)):
        dec = load_decoder(path, dev)
        scores[label] = score(dec, rows, utmos, dev)["utmos"]
        del dec
    report = {"voice": a.voice, "n": len(rows),
              la: {"path": pa, "utmos": round(float(np.mean(scores[la])), 4)},
              lb: {"path": pb, "utmos": round(float(np.mean(scores[lb])), 4)},
              f"{lb}-{la}": paired(scores[lb], scores[la]),
              "per_row": {k: rounded(v) for k, v in scores.items()}}
    print({k: v for k, v in report.items() if k != "per_row"}, flush=True)
    save_json(a.out or f"out/eval/pair-{a.voice}-{la}-{lb}.json", report)


if __name__ == "__main__":
    main()
