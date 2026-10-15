# eval

The evaluation behind the paper's tables. Every module runs from the repository root as
`python -m eval.<module> ...` (`--help` lists the options) and writes its summary and every per-row value to a JSON
file under `out/eval/` unless `--out` is given.

Shared conventions:

- Held-out rows are the 200 held-out sentences of a voice's teacher corpus, `runs/<voice>/teacher/heldout.pt`,
  written by `python -m training.gen_teacher <voice>`; "first N" always means rows 0 to N-1.
- Checkpoint paths in the examples follow `training/recipe.py` (`runs/<voice>/{text,stage1,restart,gan10,gan3,export}/`).
- Teacher conditioning ("teacher-forced"): the decoder receives Kokoro's own phoneme features, durations, F0 and
  energy for the whole sentence.
- Every decoder renders held-out row i with excitation seed `1000 + i` (`torch.manual_seed`), in every condition,
  so decoders compared on the same rows see the same random draws before their mixer.
- UTMOS is `training.utmos.Utmos` (SpeechMOS `utmos22_strong`, pinned commit). The 95% interval of a mean is
  1.96 x population SD / sqrt(N). Paired differences are the mean per-row difference with its standard error
  (`stats.py`).
- The PyTorch modules need the training dependencies (torch, torchaudio, kokoro); a CUDA GPU is used when
  available (`--device` overrides). The paper's numbers were computed on CUDA.

## decoder.py: UTMOS of decoders alone

```
python -m eval.decoder af_heart final=runs/af_heart/gan3/final.pt stage1=runs/af_heart/restart/final.pt --n 200
python -m eval.decoder af_heart final=runs/af_heart/gan3/final.pt --n 64 --distances
```

Each decoder renders the first N held-out rows teacher-forced; UTMOS per row, mean and 95% interval. The teacher's
own audio of the same rows is scored too ("teacher target"). `--distances` adds the two terms of the stage-one
objective against the teacher's audio (log-mel L1 and multi-resolution STFT) and the periodicity (normalized
autocorrelation at the F0 lag over voiced 40 ms frames) of each render and of the teacher's audio.

Paper: Tables `tab:english200` and `tab:italian` (`--n 200`), Table `tab:english64` (`--n 64 --distances`), the
teacher column of Table `tab:longgan`, the 200-row UTMOS column of Table `tab:contrastfix`, and the decoder rows of
Table `tab:blind` (`--n 16 --distances`). `--n 16` is also the 16-sentence score monitored during training
(Table `tab:ablation`, Figure `fig:chains`).

## pair.py: paired comparison of two decoders

```
python -m eval.pair if_sara gan3_5k=path/to/gan3_5k.pt final=runs/if_sara/gan3/final.pt --n 200
```

Both decoders render the same rows with the same seeds as in `decoder.py`. Reports B - A: mean per-row
difference, standard error, paired t-test, number of rows on which B scores higher, exact sign test.

Paper: the paired difference and "better" columns of Table `tab:longgan`; the comparison of the 50k and 30k stage
one in Section "Italian voices".

## e2e.py: end to end in PyTorch

```
python -m eval.e2e im_nicola final=runs/im_nicola/text/final.pt,runs/im_nicola/gan3/final.pt --n 200
```

Per text side and decoder: UTMOS end to end (the text student's rounded durations, F0, energy and phoneme features
from the token ids, then the decoder; seed `1000 + i`), UTMOS of the same decoder teacher-forced, the paired
difference end to end minus teacher-forced, and the length ratio (predicted frames over the teacher's). The
teacher's audio is scored once per voice. Text students are `training.student.TextStudent("s+mlp")` checkpoints
(key `model`).

Paper: Table `tab:e2e`.

## int8.py: exported graphs, float32 against int8

```
python -m eval.int8 if_sara runs/if_sara/export/model.onnx int8=runs/if_sara/export/model_int8.onnx --n 100
python -m eval.int8 if_sara runs/if_sara/export/model.onnx --variants text_only,decoder_only,keep3 --n 100
```

Every graph synthesizes the first N held-out sentences of the voice's language from their token ids
(`heldout.json`) at speed 1.0 with ONNX Runtime on the CPU; UTMOS per row. Each other graph is paired with the
float32 graph (other minus float32), and its size is reported. `--variants` builds partial int8 copies of the
float32 graph with `training/quantize.py`, leaving some weights in float32: `text_only` (int8 text side),
`decoder_only` (int8 decoder) and `keep3` (everything int8 except the decoder's `generator.conv_post`,
`generator.ups.1` and `generator.resblocks.4`).

ONNX Runtime's random ops ignore numpy and torch seeds, so before loading, every random op of each graph gets a
seed computed from its node name (CRC-32 modulo 2^24): the float32 graph and an int8 copy with the same node names
then draw the same excitation, and the numbers repeat from run to run. The rows run in order through one session
per graph, so a sentence whose frame count changes also shifts the draws of the sentences after it. Needs `onnx`
and `onnxruntime`, and torch for UTMOS.

Paper: the size and the int8 change in UTMOS of Table `tab:int8` (the rows "text side only", "decoder only" and
"all but 3 decoder layers" with `--variants`). The parity gate column is the export's own validation
(`runs/<voice>/export/export-validation.json`, written by `training/export.py`).

## contrast.py: harmonic contrast and HNR

```
python -m eval.contrast if_sara final=runs/if_sara/gan3/final.pt gan10=runs/if_sara/gan10/final.pt --n 40
```

The teacher's audio and each decoder's teacher-forced renders of the first N rows. F0 by YIN on 300-sample hops;
for voiced hops whose frame is within 25 dB of the waveform's 95th-percentile frame level, a 40 ms Hann window is
transformed with a 4096-point FFT, and for every harmonic k in a band the peak level within +-10% of F0 around
k F0 is compared with the mean level within +-10% of F0 around (k + 1/2) F0. The contrast per band (0-1, 1-2, 2-4,
4-6 kHz) is mean peak minus mean valley in dB; HNR is 10 log10(r / (1 - r)) from the normalized autocorrelation r
at the pitch period over 40 ms. Both are averaged over hops, then over rows.

Paper: Table `tab:contrast` and the contrast columns of Table `tab:contrastfix` (40 rows).

## wer_similarity.py: word error rate and speaker similarity

```
python -m eval.wer_similarity af_heart final=runs/af_heart/text/final.pt,runs/af_heart/gan3/final.pt --n 200
```

Three conditions on the first N rows: the teacher's audio, each decoder teacher-forced, each text side and decoder
end to end (seed `1000 + i`). WER: `openai/whisper-large-v3-turbo` (fp16 on CUDA, language forced, batch 16) on
audio resampled to 16 kHz; text and transcript lower-cased, accents and punctuation removed; word-level edit
distance summed over rows divided by the number of reference words; also the paired per-row difference from the
teacher. Speaker similarity: cosine of L2-normalized `microsoft/wavlm-base-plus-sv` x-vectors against the
teacher's audio of the same sentence (mean and standard error) and against the normalized mean x-vector of the
teacher on the first 40 training rows, which the teacher renders first because training rows are stored without
audio. Models are downloaded from the Hugging Face hub. Needs `transformers` and `scipy`.

Paper: Table `tab:wer`.

## stats.py

`ci95`, `paired` (mean difference, standard error, paired t-test with n - 1 degrees of freedom, rows better and
worse, exact two-sided sign test with ties left out) and their helpers, used by the modules above. `paired` also
returns the normal approximation of the t-test p-value, which `paper/data/stats.py` uses for the paper's
significance values.

## bench_onnx.py and heldout.json

- `bench_onnx.py`: real-time factor of exported ONNX graphs on one CPU thread, with Paradee's protocol
  (the first 8 held-out sentences, from phoneme ids). Needs `numpy` and `onnxruntime`.
- `heldout.json`: the 200 held-out sentences per language (English from WikiText-103, Italian from Wikipedia; CC BY-SA 4.0, see `training/data/LICENSE.md`), with the token ids the teacher data used.

```
python eval/bench_onnx.py eval/heldout.json 9 af_heart:ours=runs/af_heart/export/model_int8.onnx af_heart:released=path/to/released_int8.onnx
```
