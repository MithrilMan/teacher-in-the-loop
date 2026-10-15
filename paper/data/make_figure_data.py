"""Derive the small wide-format CSV files the pgfplots figures read from data/curves_raw.csv.

curves_raw.csv holds every in-training evaluation of the runs used in the paper (the `log.jsonl` that
each training run writes: UTMOS on the first 16 held-out rows, mel L1 and MR-STFT distance to the teacher
target), one row per evaluation. Run names are explained in SOURCES.md.

Standard library only. Run from the paper/ folder:  python data/make_figure_data.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    runs = {}
    with open(os.path.join(HERE, "curves_raw.csv"), newline="") as f:
        for r in csv.DictReader(f):
            runs.setdefault(r["run"], []).append((int(r["step"]), float(r["utmos16"]), float(r["mel_l1"])))
    return runs


def write(name, header, rows):
    with open(os.path.join(HERE, name), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(["nan" if v is None else v for v in r])


def wide(runs, names, offsets=None, field=1):
    """One row per step; one column per run (nan where a run has no point at that step)."""
    offsets = offsets or {}
    table = {}
    for n in names:
        for p in runs.get(n, []):
            step = p[0] + offsets.get(n, 0)
            table.setdefault(step, {})[n] = round(p[field], 4)
    return [[s] + [table[s].get(n) for n in names] for s in sorted(table)]


def chain(runs, parts):
    """Concatenate runs into one curve: parts = [(run, offset)], dropping each later run's step 0 duplicate."""
    out = []
    for i, (n, off) in enumerate(parts):
        for step, u, m in runs.get(n, []):
            if i > 0 and step == 0:
                continue
            out.append([step + off, round(u, 4), round(m, 4)])
    return out


def main():
    runs = load()
    # Figure: stage-1 ablation arms from scratch (English, 12 shards, seed 1).
    arms = ["s0-recipe", "s1-tl", "s2-tl-head", "s3-gan", "s5-tl-phase5", "s6-tl-head-gan"]
    write("fig_stage1_arms.csv", ["step", "s0", "s1", "s2", "s3", "s5", "s6"], wide(runs, arms))
    # Same arms, mel L1 to the target (shows the loss falls while UTMOS stays flat).
    write("fig_stage1_arms_mel.csv", ["step", "s0", "s1", "s2", "s3", "s5", "s6"], wide(runs, arms, field=2))
    # Figure: spectral-only fine-tune of the released decoder.
    write("fig_released_ft.csv", ["step", "utmos16", "mel_l1"],
          [[s, round(u, 4), round(m, 4)] for s, u, m in runs["ft-released-spec"]])
    # Figure: full English chain of the final recipe, 30k stage 1 -> warm restart to 50k -> GAN10 -> GAN3 30k.
    write("fig_en_chain.csv", ["step", "utmos16", "mel_l1"],
          chain(runs, [("s7-pool-30k", 0), ("s8-pool-50k", 30000), ("e-s8-g10", 50000), ("g1-en-g3-30k", 55000)]))
    # 30k branch: GAN10 -> GAN3 directly after the 30k stage 1.
    write("fig_en_chain30.csv", ["step", "utmos16", "mel_l1"],
          chain(runs, [("e-s7-g10", 30000), ("e-s7-g10-g3", 35000)]))
    # Italian chains (final recipe), same layout.
    for voice in ("sara", "nicola"):
        write(f"fig_{voice}_chain.csv", ["step", "utmos16", "mel_l1"],
              chain(runs, [(f"it-{voice}-s1", 0), (f"it-{voice}-s8", 30000), (f"it-{voice}-s8-g10", 50000),
                           (f"g1-{voice}-g3-30k", 55000)]))
    # The published 5k GAN3 from the same GAN10 decoders (the 50k recipe), drawn dashed for comparison.
    for voice, g10, g3 in (("en", "e-s8-g10", "e-s8-g10-g3"), ("sara", "it-sara-s8-g10", "it-sara-s8-g3"),
                           ("nicola", "it-nicola-s8-g10", "it-nicola-s8-g3")):
        last10 = [p for p in runs[g10] if p[0] == 5000]
        write(f"fig_{voice}_g3short.csv", ["step", "utmos16", "mel_l1"],
              [[55000, round(last10[0][1], 4), round(last10[0][2], 4)]]
              + [[s + 55000, round(u, 4), round(m, 4)] for s, u, m in runs[g3] if s > 0])


if __name__ == "__main__":
    main()
