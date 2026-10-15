# Sources of every number in the paper

Every number in the paper is in a CSV file in this folder. Each table CSV has a `source` column; the raw
curves `curves_raw.csv`, the derived `fig_*.csv` and `paired_tests.csv` are described in the file table
below. Numbers that belong to no table are in `other_numbers.csv` or in the key-value files `pool.csv`,
`upstream_reproduction_meta.csv` and `italian_context.csv`; the index at the end of this file says where
each one is. Nothing in the paper is computed from anything other than these files, the two scripts in this
folder (`make_figure_data.py`, `stats.py`) and the Paradee paper.

The experiments were run with the author's research code, from which the training and evaluation packages
of this repository (`training/`, `eval/`) were ported. The port was checked on `if_sara` against the
paper's decoder (Appendix G of the paper). The raw logs and result files of the runs are not published;
the `source` column names them so that every value can be traced to one run and one evaluation.

## Keys used in the `source` columns

| key | meaning |
|---|---|
| `P:` | The Paradee paper, Mahendrakar 2026, arXiv:2610.06817. Section or table given. |
| `R:` | A result file of one of the author's runs, named by run (`<run>/log.jsonl`, `<run>.log`) or by evaluation (`<evaluation>.json`, `<evaluation>.log`). Labels in parentheses name the entry inside the file. Run names are explained below. |
| `L:` | A run of a script in this repository's `eval/` on the author's desktop (AMD Ryzen 9 9950X3D, Windows 11, ONNX Runtime 1.30.0). |
| `code` | Where the setting is implemented in this repository's code. |
| `N` | The author's lab notebook (not published). |
| `D` | The author's internal reports (not published). |

## Run names

| name | run |
|---|---|
| `s0` … `s6` | stage-one arms from scratch for `af_heart` (Table "stage-one arms"): `s0-recipe` upstream objective, `s1-tl` teacher in the loop, `s2-tl-head` plus head losses, `s3-gan` adversarial from step 0 on the stored target, `s5-tl-phase5` phase weight 5, `s6-tl-head-gan` head losses and adversarial from step 0 |
| `s7-pool-30k`, `s8-pool-50k` | English stage one from the pool: 30,000 steps, then the 20,000-step warm restart |
| `e-s7-*`, `e-s8-*` | English adversarial stages after s7 or s8 (`g10` GAN10, `g3` GAN3) |
| `g1-s2-gan10`, `g2-gan3` | adversarial stages on top of the 10,000-step s2 decoder |
| `it-<voice>-s1`, `it-<voice>-s8` | Italian stage one, 30,000 and 50,000 steps |
| `it-<voice>-g10`, `it-<voice>-g3`, `it-<voice>-s8-g10`, `it-<voice>-s8-g3` | Italian adversarial stages after the 30k or the 50k stage one (the 30k and 50k recipes) |
| `g1-<voice>-g3-30k`, `g1*` | the 30,000-step GAN3 of the final recipe (`g1` if_sara, `g1n` im_nicola, `g1e` English) |
| `h1-sara-comb`, `h1-sara-ctrl` | the harmonic-comb test and its control (Appendix E) |
| `ft-*` | fine-tunes of the released decoder (Section 3.2 and Appendix B) |
| `final-en`, `final-en-upstream`, `final-<voice>` | 64- and 200-row evaluations of the 30k recipe and of the upstream recipe |
| `locktest*` | 200-row evaluations, raw and with the phase-locking filter, with 95 % intervals |
| `*-pair`, `pair-*` | paired 200-row comparisons (same rows, same excitation seeds), as `eval.pair` computes them |
| `*-wide-*`, `contrast-wide`, `w2n-pair`, `onnxcheck-nicola-w2` | the wider decoder (Table "wider decoder"); contrast for if_sara only |
| `onnx*`, `export-*`, `onnxcheck-*` | ONNX export and the float32 vs int8 end-to-end check |
| `paperdiag-*` | word error rate and speaker similarity |
| `seeds` | the three-seed check of the final recipe for if_sara |
| `j-s1`, `j-g10`, `j-g3`, `jeval-200` | the joint two-voice Italian decoder (stage one, GAN10, GAN3) and its paired evaluation |
| `valid-*` | evaluations of the released training package, rerun on if_sara (Appendix G) |

## Evaluation protocol shared by all `R:` numbers

- UTMOS: SpeechMOS `utmos22_strong` (source commit `ed25eacb…`, checkpoint sha256 `38aa51ab…`), on GPU.
- "16 rows": the first 16 held-out rows, logged during training. "64 rows" / "200 rows": the first 64 / all
  200 held-out rows.
- Teacher-forced ("teacher conditioning"): the decoder receives Kokoro's own durations, F0, energy and
  phoneme features for the full utterance, with excitation seed `1000 + row`, so two decoders compared on
  the same rows see the same random draws before their mixer.
- `ci95` = 1.96 × population SD / sqrt(n) of the per-row scores.
- Paired differences: mean per-row difference ± standard error; significance in `paired_tests.csv`
  (computed by `stats.py`).
- English held-out rows: the first 200 rows of the reproduction's teacher corpus. Italian: the 200 held-out
  rows of each voice's teacher corpus (`training/data/italian_sents.json`).

## Files

| file | content | origin |
|---|---|---|
| `paradee_reference.csv` | numbers quoted from the Paradee paper | `P:` |
| `objective_blindness.csv` | the two stage-one loss terms against UTMOS (16 rows) for the released and reproduced decoders | `R:baseline` |
| `released_finetune.csv` | fine-tunes of the released decoder with spectral objectives | `R:ft-*` |
| `source_sensitivity.csv` | re-draw distance; UTMOS with the harmonic source removed or replaced by noise (16 rows) | `R:sensitivity` |
| `crop_target.csv` | teacher rendered on the crop vs the stored full-utterance target | `N` (the probe printed its results; no result file) |
| `stage1_ablation.csv` | stage-one arms from scratch (16 rows) | `R:s0` … `R:s6` |
| `gan_from_s2.csv` | GAN10 / GAN3 on top of the 10k teacher-loop model | `R:g1-s2-gan10`, `R:g2-gan3` |
| `final_en_64.csv` | English, 64 rows | `R:final-en`, `R:final-en-upstream` |
| `final_en_200.csv` | English, 200 rows, raw and with the phase-locking filter | `R:locktest-200*`; three rows from `N`, because the result file holding them was overwritten by a later run |
| `phase_lock_64.csv` | phase-locking filter on 64 rows | `R:locktest` |
| `italian_200.csv` | if_sara and im_nicola, 200 rows | `R:final-<voice>`, `R:locktest-200-<voice>*`, `R:g1*-pair` |
| `gan3_long.csv` | GAN3 for 30,000 instead of 5,000 steps, all three voices: paired 200-row result and 16-row curves | `R:g1*-pair`, `R:g1-<voice>-g3-30k` |
| `paired_50k_vs_30k.csv` | 50k vs 30k stage one, paired | `R:pair-<voice>`, `R:locktest-200` |
| `end_to_end.csv` | text side + decoder, from phonemes | `R:e2e-it*`, `R:paper-e2e-*` |
| `joint_decoder.csv` | joint two-voice decoder vs single-voice decoders | `R:jeval-200` |
| `int8_export.csv` | ONNX Runtime float32 vs int8 graphs, parity gate | `R:onnx*`, `R:export-*` |
| `int8_sensitivity.csv` | PyTorch fake-quantization sensitivity | `R:qsens-sara`, `R:tqsens-sara` |
| `int8_scheme_v2.csv` | per-output-channel int8 vs the original scheme on the earlier Italian models | `D` |
| `speed.csv` | real-time factor of the int8 graphs on one CPU thread (Paradee's protocol) | `L:eval/bench_onnx.py` |
| `upstream_reproduction.csv`, `upstream_reproduction_meta.csv` | the reproduction of the upstream recipe with the unmodified upstream scripts (CPU renders, 200 rows) | `D` |
| `listening.csv` | the two blind listening checks | `N` |
| `pool.csv` | the pre-rendered teacher pool | `N`, `R:pool-<voice>`, `code training/pool.py` |
| `compute.csv` | wall-clock and step times | `R:<run>.log` (first timestamp and last step-time line); `N`, `D` where the row says so |
| `italian_context.csv` | Italian corpus, phoneme conversion, style-length dependence, upstream mixer and normalization facts | `D`, `training/data/italian_sents.json`, `training/README.md` |
| `recipe_hparams.csv` | every setting, upstream vs ours | `P:` §4.4, `code` (`training/`) |
| `harmonic_contrast.csv` | harmonic contrast per band and HNR, 40 held-out rows, teacher-forced | `R:contrast-*` |
| `wer_similarity.csv` | word error rate and speaker similarity per voice, decoder and condition | `R:paperdiag-en`, `R:paperdiag-it` |
| `seeds.csv` | if_sara, final recipe with training seeds 1, 2 and 3 | `R:locktest-200-seeds` |
| `wide_decoder.csv` | the wider decoder against the adopted one, if_sara, and im_nicola without contrast | `R:pair-wide-*`, `R:contrast-wide`, `R:onnxcheck-sara-*`, `L:eval/bench_onnx.py` |
| `other_numbers.csv` | numbers of the text that belong to no table: hardware, evaluation and parity-gate settings, the int8 schemes on the earlier models, the joint decoder's batch, the check of the released training package, Kokoro's decoder layout and a few single values | `D`, `N`, `R:`, `code`, the cited sources (per row) |
| `curves_raw.csv` | every in-training evaluation (16 rows) of the runs used | `R:<run>/log.jsonl` |
| `fig_*.csv` | figure inputs | derived from `curves_raw.csv` by `make_figure_data.py` |
| `paired_tests.csv` | z, p and sign-test values | derived by `stats.py` from the CSVs above |

## Harmonic contrast

The decoder renders the first 40 held-out rows from the teacher's own features (seed `1000 + row`). On each
output, F0 is estimated with YIN (300-sample hops). For voiced hops whose level is within 25 dB of the
utterance's 95th-percentile frame level, a 40 ms (960-sample) Hann window centred on the hop is
transformed with a 4096-point FFT. For every harmonic k in a band, the peak is the maximum level within
±10 % of F0 around k·F0 and the valley the mean level within ±10 % of F0 around (k+0.5)·F0; the hop's
contrast is mean(peaks) − mean(valleys) in dB, averaged over hops and rows. Bands 0–1, 1–2, 2–4 and 4–6 kHz.
HNR: normalized autocorrelation r at the YIN pitch period over 40 ms, 10·log10(r/(1−r)), averaged over
voiced hops. `eval.contrast` in this repository implements the measure.

The teacher row and the 50k-recipe row of `R:contrast-sara` reappear with identical values in
`R:contrast-h1` and `R:contrast-g1`: same decoder files, same 40 rows and seeds, same measure.

## Experiments in more detail

**The longer GAN3** (runs `g1*`): GAN3 from the 50k recipe's GAN10 decoder, identical to the recipe's GAN3
(batch 8, warmup 1,000, cosine decay, seed 1, whole pool, head and phase weight 1, reconstruction weight 3,
fresh critics) except for 30,000 steps instead of 5,000, with an evaluation and a generator snapshot every
2,500 steps. The pools were rendered again with the settings of the original pools (150,000 crops, seed 0;
fp32 teacher in Italian, fp16 teacher and 12 shards in English). Paired 200-row evaluation against the
recipe's 5,000-step GAN3.

**The harmonic comb** (runs `h1-*`): two 5,000-step GAN3 runs from the 50k-recipe if_sara decoder with the
recipe's GAN3 settings, one with a zero-initialized input carrying the cosine and sine of k times the phase
of the fundamental for the 48 harmonics below 11 kHz (+76,896 parameters), one without (control).

**End to end**: 200 held-out rows, text side → decoder from phoneme ids, excitation seed `1000 + row`,
paired against the same decoder teacher-forced. English text side: the `s+mlp` text side of reproduction A.
Italian text sides: the `s+mlp` text sides of the earlier Italian students.

**Export**: float32 and int8 graphs of the final decoders with the voices' text sides; end to end on the
first 100 held-out rows, same inputs and seeds.

**Speed** (`eval/bench_onnx.py`, the first 8 rows per language of `eval/heldout.json`): one intra-op and
one inter-op thread, sequential execution, CPU provider; one warm-up pass, then 5 (session 1) or 9
(session 2) timed passes; × real time = seconds of audio / seconds of compute, median and best pass. The
released graph is `paradee_int8.onnx` of Hugging Face revision `8f34b01`. Session 2 ran the released graph
first and last: 23.4× and 25.3×.

**Word error rate and speaker similarity** (`eval.wer_similarity` implements the same protocol): 200
held-out rows, three conditions (teacher audio; decoder on the teacher's features, seed `1000 + row`; text
side → decoder from phoneme ids, same seed). Whisper large-v3-turbo (`openai/whisper-large-v3-turbo`,
fp16, language forced, batch 16) on audio resampled to 16 kHz; text and transcript lower-cased, accents and
punctuation removed; WER = word-level edit distance summed over rows / reference words. Speaker
similarity: `microsoft/wavlm-base-plus-sv` x-vectors (L2-normalized), cosine to the teacher audio of the
same row (mean ± SE) and to the normalized mean of 40 training rows of the teacher. `wer_minus_teacher_*`
in the CSV is the mean ± SE of the per-row WER difference (paired).

**Seeds**: the final recipe for if_sara with training seeds 2 and 3, on the same teacher rows, crop-pool
settings and evaluation rows as the paper's run (seed 1).

**The released training package** (Appendix G): `python -m training.recipe` rerun on the paper's if_sara
teacher rows and pool, evaluated with `eval.pair`, `eval.e2e` and `eval.int8` against the paper's decoder.

**The wider decoder**: generator channels 256/128/64 and AdaIN 192 (6.54M parameters) instead of
128/64/32 and 256 (3.85M), trained with the same pool, losses, seed and stages. GFLOP per second of audio
counted from the decoder definition: 3.36 (adopted) and 12.57 (wider). Speed in one session of
`eval/bench_onnx.py` (5 passes): adopted int8 27.5×, wider int8 12.3×, adopted fp32 28.3×, wider fp32 12.5×.

## Numbers in the text that are not in a table

| number in the paper | value | source |
|---|---|---|
| Paradee stage-one decoder alone (Table 5) | 2.98 (full 3.00) | `P:` Table 5 |
| Paradee final decoder / full / filtered | 4.37 / 4.39 / 4.41 | `P:` Table 5 |
| released decoder vs reproduction differ by ~4 % in the stage-one loss | (0.5198+0.9758)/(0.5504+0.8894) − 1 = 3.9 % | `objective_blindness.csv` |
| teacher on the crop misses the stored target | mel L1 1.24 | `crop_target.csv` |
| reproductions of the upstream recipe, full chain (64 rows) | 1.866 / 2.848 | `final_en_64.csv` |
| same, 200 rows | 1.94 / 2.89 | `final_en_200.csv` |
| same, CPU renders, 200 rows, teacher-conditioned | 1.9388 (released 4.3712) | `upstream_reproduction.csv` |
| stage one of the reproduction (200 rows, CPU) | 1.2349 | `upstream_reproduction.csv` |
| teacher loop + head, 10k | 2.72 (upstream arm 1.23) | `stage1_ablation.csv` |
| GAN10 5k / GAN3 2k on top | 3.59 / 3.84 | `gan_from_s2.csv`, `curves_raw.csv` (`g2-gan3` step 2000 = 3.840) |
| pool step time | 0.27 s vs 0.57–0.65 s | `pool.csv` |
| 30k stage one → GAN10 → GAN3, 64 rows | 4.365 (stage one 3.934) | `final_en_64.csv` |
| 50k recipe, 200 rows | 4.414 ± 0.008 (30k 4.372; stage one 4.131 vs 3.930) | `final_en_200.csv` |
| GAN10 → GAN3 vs GAN3 alone, 200 rows | +0.014 | `final_en_200.csv` (4.372 − 4.358) |
| Italian 50k | if_sara 3.752, im_nicola 3.601; paired +0.041 ± 0.005 / +0.026 ± 0.003 | `italian_200.csv`, `paired_50k_vs_30k.csv` |
| end to end, 50k recipe | if_sara 3.766, im_nicola 3.609, English 4.424 | `end_to_end.csv` |
| joint decoder | −0.089 ± 0.008 / −0.086 ± 0.005 | `joint_decoder.csv` |
| int8 ONNX | ≤ 0.024 loss over all graphs; ≤ 0.010 for the final recipe | `int8_export.csv` |
| GPU memory per adversarial job | ~6.1–6.2 GB | `compute.csv` (`N`) |
| hardware | modified RTX 4080 SUPER, 32,760 MiB; 8 vCPU Xeon Gold 6338; 43 GiB RAM; each job limited to 4 CPU cores and 20 GB RAM; usually one to three of the author's jobs on the GPU at once | `other_numbers.csv` (`D`, `N`) |
| Italian corpus | 12,000 + 200 sentences from 1,203 Italian Wikipedia articles (1,312 scanned), article-disjoint | `training/data/italian_sents.json` |
| if_sara teacher audio | 33.2495 h | `italian_context.csv` |
| Italian voices graded C | — | Kokoro `VOICES.md` (cited) |
| style vector change across held-out lengths (relative RMS) | 6.50 % / 6.85 % / 7.97 % (paper: 6.5–8.0 %) | `italian_context.csv` |
| issue #3 independent report | ~1.2 after reconstruction | `other_numbers.csv` (the public GitHub issue, cited) |
| Paradee int8 file: paper vs released | 8.45 MB in the paper (§4.6); the released `paradee_int8.onnx` (Hugging Face revision `8f34b01`) is 9,037,971 bytes = 9.04 MB | `paradee_reference.csv` (`P:` §4.6); `other_numbers.csv` (size of the downloaded file) |
| GAN3 30k vs 5k, if_sara, paired | 3.8819 vs 3.7524; +0.1295 ± 0.007; 187/200; z = 18.5 | `gan3_long.csv`, `paired_tests.csv` |
| GAN3 30k vs 5k, English, paired | 4.4670 vs 4.4144; +0.0526 ± 0.0024; 197/200; z = 21.9 | `gan3_long.csv`, `paired_tests.csv` |
| GAN3 30k vs 5k, im_nicola, paired | 3.6959 vs 3.6014; +0.0944 ± 0.0033; 197/200; z = 28.6 | `gan3_long.csv`, `paired_tests.csv` |
| GAN3 30k, 16 rows during training | if_sara 3.7755 (0), 3.859 (5k), 3.9507 (10k), 3.96 (12.5k), then 3.9357–3.9846 to 30k; im_nicola 3.6005 (0), 3.6738 (5k), 3.7434 (15k), 3.7494–3.7624 (17.5k–30k); English 4.3861 (0), 4.407 (5k), 4.4446 (10k), 4.4592 (20k), 4.4651 (30k) | `gan3_long.csv` |
| adversarial stages after the 50k stage one | English 4.4670 − 4.1311 = 0.336 (5k: 0.283); if_sara 0.137 (0.008); im_nicola 3.6959 − 3.4257 = 0.270 (0.176) | `final_en_200.csv`, `italian_200.csv`, `gan3_long.csv` |
| gap to the teacher, final recipe | English 4.5211 − 4.4670 = 0.054 (50k: 0.107); if_sara 4.0216 − 3.8819 = 0.14 (0.27); im_nicola 3.8666 − 3.6959 = 0.171 (0.265) | `gan3_long.csv`, `italian_200.csv` |
| final English vs released, same rows | 4.4670 − 4.3708 = 0.096 | `gan3_long.csv`, `final_en_200.csv` |
| GAN3 wall clock | 5k: 51 min (if_sara, im_nicola), 64 min (English); 30k: 5 h 25 min, 6 h 30 min, 6 h 41 min | `compute.csv` |
| English chain wall clock (Appendix G) | stage one 30k 2 h 49 min, +20k 1 h 20 min, GAN10 50 min (22:53:56 to 23:44:14), GAN3 5k 64 min | `compute.csv` |
| learning rate at step 5,000 of the 30k GAN3 | 93 % of peak: min(1, s/1000) · 0.5 (1 + cos(π s / 30000)) at s = 5000; the 5k GAN3 reaches 0 at the same step | `other_numbers.csv` (`code training/train_decoder.py`, schedule) |
| GAN3 30k closes about 30 % of the 2–4 kHz gap (if_sara) | (7.69 − 5.95) / (11.81 − 5.95) = 1.74 / 5.86 = 0.297 | `harmonic_contrast.csv` |
| GAN3 30k, other bands (if_sara) | 1–2 kHz 15.70 → 16.47; 4–6 kHz 3.57 → 3.99; HNR 12.81 → 12.83 | `harmonic_contrast.csv` |
| English contrast, GAN3 30k | 2–4 kHz 4.18 → 5.45 dB (4.33 / 4.88 / 5.35 at 5k / 10k / 20k); (5.45 − 4.18) / (10.52 − 4.18) = 0.20; 1–2 kHz 14.17 → 15.01 | `harmonic_contrast.csv` |
| im_nicola teacher contrast | 13.4 dB (0–1 kHz), 2.8 dB (2–4 kHz); not interpreted | `harmonic_contrast.csv` |
| comb vs control | 2–4 kHz 6.07 vs 6.01 dB; largest band difference 0.11 dB (1–2 kHz 15.99 vs 15.88); UTMOS 3.7865 ± 0.0262 vs 3.7678 ± 0.0260; paired +0.0187 ± 0.0038, 137 of 200 better | `harmonic_contrast.csv`, `italian_200.csv`, `paired_tests.csv` |
| comb input size | 77k parameters (76,896) | `other_numbers.csv` (`R:h1-sara-comb.log`) |
| end to end, final recipe | English 4.474 (tf 4.467, +0.007 ± 0.002), if_sara 3.895 (+0.013 ± 0.008), im_nicola 3.700 (+0.004 ± 0.005) | `end_to_end.csv` |
| end to end, released decoder with reproduction A's text side | 4.391 (tf 4.371, +0.020 ± 0.004); complete released model 4.395 | `end_to_end.csv`, `upstream_reproduction.csv` |
| int8, final recipe | English −0.0061 ± 0.0008, if_sara −0.0072 ± 0.0109, im_nicola −0.0099 ± 0.0049 | `int8_export.csv` |
| speed, final recipe int8 | 24.3–26.3× (English), 24.7–25.5× (if_sara), 24.9× (im_nicola); released int8 23.4–25.3×; audio 61.2–71.6 s | `speed.csv` |
| Paradee speed | 17.8× real time, Apple M4 Pro, one thread | `P:` Table 3 |
| teacher WER | af_heart 3.32 %, if_sara 4.09 %, im_nicola 4.45 % | `wer_similarity.csv` |
| students' WER range | 3.12 % (English 50k, end to end) to 4.33 % (im_nicola final) | `wer_similarity.csv` |
| largest per-row WER difference from the teacher | im_nicola 50k teacher-forced −0.0049 ± 0.0029 (1.7 SE) | `wer_similarity.csv` |
| same-sentence similarity, teacher-forced | English final 0.9925, 50k 0.9912, released 0.9853; if_sara 50k 0.9893 → final 0.9956; im_nicola 0.9942 / 0.9945 | `wer_similarity.csv` |
| similarity drop end to end vs teacher-forced | at most 0.0024 (English final 0.9925 → 0.9901) | `wer_similarity.csv` |
| stage one (50k) by seed | 3.745, 3.431, 3.403 | `seeds.csv` |
| final recipe by seed | 3.882, 3.881, 3.883 (95 % CI ±0.023 to ±0.025) | `seeds.csv` |
| release package vs paper decoder (if_sara, paired, 200 rows) | −0.004 ± 0.007, better on 102 of 200; 3.878 vs 3.882 | `other_numbers.csv` (`R:valid-pair`) |
| release package end to end | 3.883 vs paper 3.894 in the same evaluation (`end_to_end.csv` has 3.8951 for the paper's decoder, from a separate evaluation) | `other_numbers.csv` (`R:valid-e2e`) |
| release package int8 vs its float32 graph (100 rows) | −0.030 ± 0.009 (paper's graph −0.007 ± 0.011) | `other_numbers.csv` (`R:valid-int8`), `int8_export.csv` |
| wider decoder | if_sara +0.133 ± 0.008 (50k) and +0.123 ± 0.007 (final), 182/200 each; 2–4 kHz 8.19 and 10.49 dB; 77 % of the gap = (10.49 − 5.95) / (11.81 − 5.95); im_nicola 50k +0.030 ± 0.004, 143/200 | `wide_decoder.csv` |
| Kokoro's decoder (Section 2) | frames of 600 samples (25 ms at 24 kHz), F0 and N at two values per frame; 512-channel phoneme features; 256-dimensional style vector; fundamental + 8 overtones; upsampling 60; `conv_post` 22 channels; iSTFT FFT 20, hop 5, 4,800 frames/s | `other_numbers.csv` (Kokoro code and `config.json`, cited) |
| GPU excitation STFT (Appendix A) | phase equal to the CPU version except 6 values per million on the branch cut; teacher output log-mel L1 0.001; step twice as fast | `pool.csv` |
| reproduction A below the released model (CPU renders) | on 200 of 200 sentences in each of the four comparisons | `upstream_reproduction_meta.csv` |
| GAN stages of reproduction A (CPU renders) | raise the score over stage one on 200 of 200 sentences on both paths, by +0.704 (teacher-conditioned) and +0.677 (full) | `upstream_reproduction_meta.csv` |
| reproduction B seeds | 42 (Python, NumPy, PyTorch) | `upstream_reproduction_meta.csv` |
| numerics (Appendix B) | spectral gradient 25 % from CPU with TF32 convolutions, 0.5 % with TF32 disabled (25.364 %, 0.5244 %); about 95 % (94.664 %) of the waveform-gradient difference from values between 1e-5 and 1e-4; paired continuation of 2,000 steps, 2.898 with TF32 and 2.904 without on 8 sentences | `upstream_reproduction_meta.csv` |
| s3 step time | 1.4 s per step | `compute.csv` |
| log-mel L1 and periodicity settings (Appendix C) | FFT 1024, hop 256, 80 bands; periodicity over voiced 40 ms frames | `recipe_hparams.csv`, `other_numbers.csv` |
| int8 scheme v2 on the earlier models (Appendix D.7) | closer to float32 in 123 of 127 chunks with unchanged durations, pooled over five earlier models; graph 9.06 → 9.25 MB; one 8.85-second sentence 395 → 410 ms with two CPU threads | `other_numbers.csv` (`D`) |
| parity gate | at most 2 % of durations changed and at most 1.5 × the re-seeded float32 distance; 3 built-in cases (546 tokens), of which 1 (if_sara) or 2 (im_nicola) keep every duration | `other_numbers.csv` |
| joint decoder batch | 16 crops per voice per step in stage one, 8 per voice in the adversarial stages | `other_numbers.csv` (`N`) |
