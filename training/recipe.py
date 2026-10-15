"""The paper's final recipe for one voice, step by step; finished steps are skipped, so it can be rerun after an
interruption (the training steps also resume from their last checkpoint).

  python -m training.recipe VOICE [--utmos] [--dry-run]

  1. teacher corpus            gen_teacher
  2. text side                 train_text, 8,000 steps
  3. teacher pool              pool, 150,000 crops
  4. stage one                 train_decoder, 30,000 steps from scratch, batch 16, lr 2e-4
  5. warm restart              20,000 more steps, peak lr 1e-4
  6. GAN10                     5,000 steps, batch 8, reconstruction weight 10
  7. GAN3                      30,000 steps, batch 8, reconstruction weight 3
  8. export                    ONNX float32 and int8

English reproduces the paper's setting: decoder stages on the first 5,799 training sentences (12 of Paradee's 24
shards) and the pool rendered by the teacher under fp16 autocast. Italian uses all 12,000 sentences and the
float32 teacher.
"""
import argparse
import os
import subprocess
import sys

from .common import VOICES, run_dir


def steps(voice, utmos):
    en = VOICES[voice] == "en"
    rows = ["--train-rows", "5799"] if en else []
    ev = ["--utmos"] if utmos else []
    r = run_dir(voice)
    return [
        ("teacher corpus", os.path.join(r, "teacher", "done"), ["gen_teacher", voice]),
        ("text side", os.path.join(r, "text", "final.pt"), ["train_text", voice, "--steps", "8000"]),
        ("teacher pool", os.path.join(r, "pool", "meta.json"), ["pool", voice, "--crops", "150000", "--seed", "0", *rows]
         + (["--teacher-fp16"] if en else [])),
        ("stage one", os.path.join(r, "stage1", "final.pt"),
         ["train_decoder", voice, "stage1", "--init", "scratch", "--steps", "30000", "--bs", "16", "--every", "2500", *rows, *ev]),
        ("warm restart", os.path.join(r, "restart", "final.pt"),
         ["train_decoder", voice, "restart", "--init", os.path.join(r, "stage1", "final.pt"), "--steps", "20000", "--lr", "1e-4",
          "--bs", "16", "--every", "2500", *rows, *ev]),
        ("GAN10", os.path.join(r, "gan10", "final.pt"),
         ["train_decoder", voice, "gan10", "--init", os.path.join(r, "restart", "final.pt"), "--steps", "5000", "--bs", "8",
          "--gan", "--mel-weight", "10", "--every", "1000", *rows, *ev]),
        ("GAN3", os.path.join(r, "gan3", "final.pt"),
         ["train_decoder", voice, "gan3", "--init", os.path.join(r, "gan10", "final.pt"), "--steps", "30000", "--bs", "8",
          "--gan", "--mel-weight", "3", "--every", "2500", *rows, *ev]),
        ("export", os.path.join(r, "export", "export-validation.json"), ["export", voice]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voice", choices=sorted(VOICES))
    ap.add_argument("--utmos", action="store_true", help="score the 16 tracked held-out sentences with UTMOS during training")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args()
    for name, done, cmd in steps(a.voice, a.utmos):
        line = [sys.executable, "-m", f"training.{cmd[0]}", *cmd[1:]]
        if os.path.exists(done):
            print(f"[{name}] done", flush=True)
            continue
        print(f"[{name}] " + " ".join(line[1:]), flush=True)
        if a.dry_run:
            continue
        subprocess.run(line, check=True)
        if name == "teacher corpus":
            open(done, "w").close()


if __name__ == "__main__":
    main()
