# Derived from Paradee's training/scripts/export_paradee.py (Apache-2.0); see NOTICE.
"""Join the text side and the decoder into one ONNX graph, store an int8 copy, and check both on the CPU.

  python -m training.export VOICE [--text runs/VOICE/text/final.pt] [--decoder runs/VOICE/gan3/final.pt]

Graph: input_ids int64 [1, T] (with the padding token 0 at both ends) and speed float32 [1] -> waveform float32
[1, N] at 24 kHz. No post-filter. The int8 copy stores large weights as int8 with one scale per output channel
(quantize.py); arithmetic stays float32. Writes runs/VOICE/export/{model.onnx,model_int8.onnx,export-validation.json}.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from .common import SR, run_dir
from .student import TextStudent
from .student_decoder import load_student_decoder


def strip(m):
    for x in m.modules():
        try:
            remove_weight_norm(x)
        except Exception:
            pass
        if is_parametrized(x, "weight"):
            remove_parametrizations(x, "weight", leave_parametrized=True)


class ConvSTFT(nn.Module):
    """An exportable STFT that matches torch.stft / torch.istft, which the decoder was trained with. Kokoro's
    CustomSTFT differs from them: replicate instead of reflect padding, float noise in the DC and Nyquist imaginary
    parts that flips their phase between +pi and -pi, and an inverse that neither halves DC/Nyquist nor divides by
    the window overlap. Upstream Paradee measured about 0.5 UTMOS lost with it."""

    def __init__(self, n, hop):
        super().__init__()
        self.n, self.hop = n, hop
        w = torch.hann_window(n)
        k = torch.arange(n // 2 + 1).float()[:, None]
        ang = 2 * torch.pi * k * torch.arange(n).float() / n
        re, im = torch.cos(ang) * w, -torch.sin(ang) * w
        im[0] = 0
        im[-1] = 0
        self.register_buffer("fwd", torch.cat([re, im])[:, None])
        c = torch.full((n // 2 + 1, 1), 2.0)
        c[0] = c[-1] = 1
        self.register_buffer("inv", (torch.cat([torch.cos(ang) * c, -torch.sin(ang) * c]) / n * w)[:, None])
        self.register_buffer("w2", (w**2)[None, None])

    def transform(self, x):
        y = F.conv1d(F.pad(x[:, None], (self.n // 2, self.n // 2), mode="reflect"), self.fwd, stride=self.hop)
        K = self.n // 2 + 1
        re, im = y[:, :K], y[:, K:]
        ph = torch.where((im == 0) & (re < 0), torch.full_like(re, torch.pi), torch.atan2(im, re))  # torch.angle gives +pi there
        return torch.sqrt(re**2 + im**2), ph

    def inverse(self, mag, phase):
        z = torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], 1)
        y = F.conv_transpose1d(z, self.inv, stride=self.hop)[:, 0]
        env = F.conv_transpose1d(torch.ones(1, 1, mag.shape[-1]), self.w2, stride=self.hop)[:, 0]
        return (y / env.clamp_min(1e-8))[:, self.n // 2 : -(self.n // 2)][:, None]


class Speaker(nn.Module):
    """Text student + decoder for one unpadded sentence, written without masks or packed sequences."""

    def __init__(self, ts, ds):
        super().__init__()
        self.ts, self.ds = ts, ds

    def text(self, ids):
        t = self.ts
        p, s = t.predictor, t.style
        x = t.bert_encoder(t.bert(ids, attention_mask=torch.ones_like(ids))).transpose(-1, -2)
        sx = s[:, :, None].expand(-1, -1, ids.shape[1])
        x = torch.cat([x, sx], 1)
        for b in p.text_encoder.lstms:
            if isinstance(b, nn.LSTM):
                x = b(x.transpose(-1, -2))[0].transpose(-1, -2)
            else:
                x = torch.cat([b(x.transpose(-1, -2), s).transpose(-1, -2), sx], 1)
        d = x.transpose(-1, -2)  # [1, T, hidden + style]
        dur = torch.sigmoid(p.duration_proj(p.lstm(d)[0])).sum(-1)[0]  # [T]
        e = t.text_encoder
        h = e.embedding(ids).transpose(1, 2)
        for c in e.cnn:
            h = c(h)
        asr_tok = t.asr_proj(e.lstm(h.transpose(1, 2))[0]).transpose(-1, -2)  # [1, 512, T]
        return d, dur, asr_tok

    def forward(self, input_ids, speed):
        d, dur, asr_tok = self.text(input_ids)
        n = torch.clamp(torch.round(dur / speed), min=1).long()
        end = torch.cumsum(n, 0)
        start = end - n
        f = torch.arange(end[-1])[None]
        aln = ((f >= start[:, None]) & (f < end[:, None])).float()[None]  # [1, T, F]
        F0, N = self.ts.predictor.F0Ntrain(d.transpose(-1, -2) @ aln, self.ts.style)
        return self.ds(asr_tok @ aln, F0, N).squeeze(1)


def load(text, decoder):
    ts = TextStudent("s+mlp").eval()
    ts.load_state_dict(torch.load(text, map_location="cpu", weights_only=False)["model"])
    ds = load_student_decoder(decoder)
    strip(ts)
    strip(ds)
    ds.generator.stft = ConvSTFT(ds.generator.stft.filter_length, ds.generator.stft.hop_length)  # parameter-free
    return Speaker(ts, ds).eval()


def check(path, cases):
    """Finite, non-silent audio on the CPU provider, getting shorter as speed rises; real-time factor."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    total = audio = 0.0
    results = []
    for ids in cases:
        n = {}
        for speed in (0.8, 1.0, 1.2):
            t = time.perf_counter()
            wav = sess.run(["waveform"], {"input_ids": ids, "speed": np.array([speed], np.float32)})[0]
            dt = time.perf_counter() - t
            if wav.ndim != 2 or wav.shape[0] != 1 or not wav.shape[1] or not np.isfinite(wav).all() or not np.abs(wav).max():
                raise AssertionError(f"{path}: bad output {wav.shape} for {ids.shape[1]} tokens at speed {speed}")
            n[speed] = wav.shape[1]
            total, audio = total + dt, audio + wav.shape[1] / SR
            results.append({"tokens": int(ids.shape[1]), "speed": speed, "samples": int(wav.shape[1]), "seconds": round(dt, 4)})
        if not n[0.8] > n[1.0] > n[1.2]:
            raise AssertionError(f"{path}: speed did not change the duration as expected: {n}")
    print(f"{os.path.basename(path)}: {audio / total:.1f}x real time on one thread", flush=True)
    return {"bytes": os.path.getsize(path), "x_realtime": round(audio / total, 1), "cases": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    ap.add_argument("--text", default=None, help="default: runs/VOICE/text/final.pt")
    ap.add_argument("--decoder", default=None, help="default: runs/VOICE/gan3/final.pt")
    a = ap.parse_args()
    import onnx

    from .quantize import parity_failures, parity_report, quantize_model

    root = run_dir(a.voice)
    out = os.path.join(root, "export")
    os.makedirs(out, exist_ok=True)
    m = load(a.text or os.path.join(root, "text", "final.pt"), a.decoder or os.path.join(root, "gan3", "final.pt"))
    held = torch.load(os.path.join(root, "teacher", "heldout.pt"), weights_only=False)[:2]

    path = os.path.join(out, "model.onnx")
    torch.manual_seed(0)
    torch.onnx.export(m, (held[1]["ids"][None], torch.ones(1)), path, input_names=["input_ids", "speed"], output_names=["waveform"],
                      dynamic_axes={"input_ids": {1: "tokens"}, "waveform": {1: "samples"}}, opset_version=17, dynamo=False)
    g = onnx.load(path)
    onnx.checker.check_model(g, full_check=True)
    q, quant = quantize_model(g)
    qpath = os.path.join(out, "model_int8.onnx")
    onnx.save(q, qpath)
    print(f"float32 {os.path.getsize(path) / 1e6:.1f} MB, int8 {os.path.getsize(qpath) / 1e6:.2f} MB "
          f"({quant['int8_tensors']} tensors, max relative error {quant['max_relative_l2_error']:.4f})", flush=True)

    base = held[0]["ids"][1:-1]
    cases = [torch.cat([torch.zeros(1, dtype=torch.long), base.repeat(n // len(base) + 1)[:n], torch.zeros(1, dtype=torch.long)])[None].numpy()
             for n in (12, 128, 400)]
    report = {"voice": a.voice, "models": {os.path.basename(p): check(p, cases) for p in (path, qpath)}}
    parity = parity_report(g, q, [c[0].tolist() for c in cases])
    failures = parity_failures(parity)
    report["int8_parity"] = {"scheme": quant["scheme"], "max_relative_l2_error": quant["max_relative_l2_error"],
                             **{k: v for k, v in parity.items() if k != "details"}, "gate_failures": failures}
    print(f"int8 parity: {parity['duration_mismatched_tokens']}/{parity['tokens']} token durations changed, median log-spectral "
          f"distance {parity['median_log_spectral_distance_db']} dB (float32 reseeded: {parity['baseline_median_log_spectral_distance_db']:.2f} dB)"
          + (" -- gate not met: " + "; ".join(failures) + "; check the int8 graph end to end (eval)" if failures else ""), flush=True)
    json.dump(report, open(os.path.join(out, "export-validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
