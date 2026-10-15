# training

The paper's recipe for one voice, from Kokoro-82M to an int8 ONNX graph. Run everything from the repository root
in an environment with the `train` extra:

```bash
pip install -e ".[train]"
python -m training.recipe if_sara
```

The exact package versions of the paper's environment (Python 3.12.15, torch 2.14.0 with CUDA 13.0) are pinned in
`requirements-lock.txt`; its header shows how to install from it.

`training.recipe` runs the eight steps below in order and skips the ones already finished; every training step
also resumes from its last checkpoint. Outputs go to `runs/<voice>/`. A CUDA GPU is needed for the teacher pool
and the decoder; the paper's runs used one 32 GB GPU shared with other jobs.

| Step | Module | Paper setting | Output |
|---|---|---|---|
| 1. Teacher corpus | `gen_teacher` | Kokoro-82M `f3ff357`, 200 held-out sentences | `teacher/` |
| 2. Text side | `train_text` | Paradee's `s+mlp` student, 8,000 steps, batch 32 | `text/` |
| 3. Teacher pool | `pool` | 150,000 random 64-frame crops | `pool/` (about 62 GB) |
| 4. Stage one | `train_decoder` | 30,000 steps from scratch, batch 16, lr 2e-4, warmup 1,000 + cosine | `stage1/` |
| 5. Warm restart | `train_decoder` | 20,000 steps, peak lr 1e-4 | `restart/` |
| 6. GAN10 | `train_decoder --gan` | 5,000 steps, batch 8, reconstruction weight 10 | `gan10/` |
| 7. GAN3 | `train_decoder --gan` | 30,000 steps, batch 8, reconstruction weight 3 | `gan3/` |
| 8. Export | `export` | float32 graph and int8 copy, checked on the CPU | `export/` |

The voices are `af_heart` (English), `if_sara` and `im_nicola` (Italian). For English the recipe reproduces the
paper's setting: the decoder trains on the first 5,799 training sentences, and the pool is rendered by the
teacher under fp16 autocast. Each step can also be run on its own, e.g.
`python -m training.train_decoder if_sara gan3 --init runs/if_sara/gan10/final.pt --steps 30000 --bs 8 --gan --mel-weight 3`;
`python -m training.recipe VOICE --dry-run` prints every command.

## What differs from Paradee

The models are Paradee's, unchanged: `student.py` (text side) and `discriminators.py` are upstream's files, and
`student_decoder.py` keeps upstream's decoder with only preset A. The changes are in how the decoder is trained
and exported (paper, Sections 3 and 4):

| | Paradee | Here |
|---|---|---|
| Decoder target | the teacher's waveform of the whole sentence, cropped | the teacher run on the crop itself, with the excitation the student receives (`pool.py`) |
| Loss, stage one | log-mel L1 + multi-resolution STFT | the same, plus L1 on the teacher's iSTFT head: log-magnitude and sin(phase) |
| Source module | randomly initialized, never trained | the teacher's, copied in and frozen |
| Stage one | 50,000 steps | 30,000 steps, then a warm restart of 20,000 |
| GAN3 | 5,000 steps on the first 8 shards | 30,000 steps on the same sentences as stage one |
| Post-filter | phase lock, 2–8 kHz | none |
| Int8 storage | one scale per tensor row along axis 0 | one scale per output channel; biases, norms and windows in float32 (`quantize.py`) |
| Phonemes | Misaki (English) | the same, plus Misaki's eSpeak NG backend for Italian (`titl.phonemes`) |

## Data

`data/wikitext_sents.json` is Paradee's list of 12,000 WikiText-103 sentences; the first 200 accepted ones are held
out. `data/italian_sents.json` holds the 200 held-out and 12,000 training sentences drawn from 1,203 articles of
the Italian Wikipedia (snapshot `20231101.it` of `wikimedia/wikipedia`), with no article in both splits; each
sentence carries the URL of its article. Both lists are under CC BY-SA 4.0, not Apache-2.0; attribution in
`data/LICENSE.md`.
