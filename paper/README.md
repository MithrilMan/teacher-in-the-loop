# Paper: Teacher in the Loop

**Teacher in the Loop: Reliable Distillation of Kokoro-82M into 8M-Parameter Single-Voice Students, in
English and Italian.** Fabio Angela, 2026. A follow-up to Paradee (Mahendrakar 2026, arXiv:2610.06817).

- `Teacher-in-the-Loop-Angela-2026.pdf`: the paper (Zenodo, DOI 10.5281/zenodo.23400040). The LaTeX source will be on arXiv.
- `data/`: every number in the paper, in CSV files, with its source. The paper calls this folder `data/`.
  Read `data/SOURCES.md` first: it says which run and evaluation each number comes from and how the
  evaluations were made.

The figures are drawn from `data/fig_*.csv`, which `data/make_figure_data.py` derives from the in-training
evaluations in `data/curves_raw.csv`; `data/stats.py` computes the paired tests in `data/paired_tests.csv`.
Both use only the Python standard library:

```sh
python paper/data/make_figure_data.py
python paper/data/stats.py
```
