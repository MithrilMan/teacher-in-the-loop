"""Significance of the paired comparisons reported in the paper, from the means and standard errors in data/*.csv.

Paired comparisons (same held-out rows, same excitation seeds) are summarised by the mean per-row
difference d and its standard error SE = sd(d)/sqrt(n), as eval.pair, eval.e2e and eval.int8 in this
repository report them. The test statistic of the paired t-test (Student, 1908) is t = d/SE;
with n = 100 or 200 rows (99 or 199 degrees of freedom) we use the normal approximation for the two-sided
p-value. Where the per-row win count is known we also give the exact two-sided sign test (Dixon and Mood,
1946). The unpaired comparisons (English; the if_sara harmonic-comb runs) use the 95% confidence
half-widths of the two means (SE = ci95 / 1.96), which ignores the pairing and is therefore conservative.

Standard library only. Run from the paper/ folder:  python data/stats.py   (writes data/paired_tests.csv)
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def p_normal(z):
    return math.erfc(abs(z) / math.sqrt(2))


def sign_test(k, n):
    """Exact two-sided binomial test of k successes in n trials at p = 0.5."""
    tail = sum(math.comb(n, i) for i in range(max(k, n - k), n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fmt_p(p):
    return "<1e-6" if p < 1e-6 else f"{p:.3g}"


ROWS = [
    # label, n, d, se, wins (or None), source
    ("if_sara 50k vs 30k recipe (paired)", 200, 0.0407, 0.0048, 153, "paired_50k_vs_30k.csv"),
    ("im_nicola 50k vs 30k recipe (paired)", 200, 0.0262, 0.0026, 158, "paired_50k_vs_30k.csv"),
    ("if_sara joint minus single decoder", 200, -0.0894, 0.0081, None, "joint_decoder.csv"),
    ("im_nicola joint minus single decoder", 200, -0.0862, 0.0046, None, "joint_decoder.csv"),
    ("af_heart phase-lock gain, released", 200, 0.0123, 0.0031, None, "final_en_200.csv"),
    ("af_heart phase-lock gain, ours 50k", 200, -0.0347, 0.0025, None, "final_en_200.csv"),
    ("if_sara phase-lock gain, ours 50k", 200, -0.1909, 0.0054, None, "italian_200.csv"),
    ("im_nicola phase-lock gain, ours 50k", 200, -0.0790, 0.0026, None, "italian_200.csv"),
    ("af_heart phase-lock gain, final recipe", 200, -0.0350, 0.0016, None, "final_en_200.csv"),
    ("if_sara phase-lock gain, final recipe", 200, -0.2216, 0.0049, None, "italian_200.csv"),
    ("im_nicola phase-lock gain, final recipe", 200, -0.0912, 0.0029, None, "italian_200.csv"),
    ("if_sara end to end minus teacher-forced (50k)", 200, 0.0133, 0.0087, None, "end_to_end.csv"),
    ("im_nicola end to end minus teacher-forced (50k)", 200, 0.0074, 0.0054, None, "end_to_end.csv"),
    ("af_heart end to end minus teacher-forced (50k)", 200, 0.0097, 0.0031, None, "end_to_end.csv"),
    ("af_heart end to end minus teacher-forced (released decoder)", 200, 0.0202, 0.0039, None, "end_to_end.csv"),
    ("af_heart end to end minus teacher-forced (final recipe)", 200, 0.0070, 0.0020, None, "end_to_end.csv"),
    ("if_sara end to end minus teacher-forced (final recipe)", 200, 0.0132, 0.0077, None, "end_to_end.csv"),
    ("im_nicola end to end minus teacher-forced (final recipe)", 200, 0.0040, 0.0051, None, "end_to_end.csv"),
    ("if_sara INT8 minus FP32 ONNX (50k)", 100, -0.0004, 0.0105, None, "int8_export.csv"),
    ("im_nicola INT8 minus FP32 ONNX (50k)", 100, -0.0078, 0.0053, None, "int8_export.csv"),
    ("af_heart INT8 minus FP32 ONNX (50k)", 100, 0.0061, 0.0022, None, "int8_export.csv"),
    ("af_heart INT8 minus FP32 ONNX (final recipe)", 100, -0.0061, 0.0008, None, "int8_export.csv"),
    ("if_sara INT8 minus FP32 ONNX (final recipe)", 100, -0.0072, 0.0109, None, "int8_export.csv"),
    ("im_nicola INT8 minus FP32 ONNX (final recipe)", 100, -0.0099, 0.0049, None, "int8_export.csv"),
    ("if_sara INT8 minus FP32 ONNX (30k, all INT8)", 100, -0.0235, 0.0105, None, "int8_export.csv"),
    ("if_sara INT8 minus FP32 ONNX (30k, decoder only)", 100, -0.0108, 0.0014, None, "int8_export.csv"),
    ("af_heart GAN3 30k minus GAN3 5k (paired)", 200, 0.0526, 0.0024, 197, "gan3_long.csv"),
    ("if_sara GAN3 30k minus GAN3 5k (paired)", 200, 0.1295, 0.0070, 187, "gan3_long.csv"),
    ("im_nicola GAN3 30k minus GAN3 5k (paired)", 200, 0.0944, 0.0033, 197, "gan3_long.csv"),
    ("if_sara H-1 comb minus control (paired)", 200, 0.0187, 0.0038, 137, "R:pair-h1.log"),
]

UNPAIRED = [
    # label, mean_a, ci_a, mean_b, ci_b, source
    ("af_heart ours 50k minus ours 30k (unpaired, conservative)", 4.4144, 0.0081, 4.3719, 0.0116, "final_en_200.csv"),
    ("af_heart ours 50k minus released (unpaired, conservative)", 4.4144, 0.0081, 4.3708, 0.0111, "final_en_200.csv"),
    ("af_heart ours 30k minus released (unpaired, conservative)", 4.3719, 0.0116, 4.3708, 0.0111, "final_en_200.csv"),
    ("af_heart final recipe minus released (unpaired, conservative)", 4.4670, 0.0055, 4.3708, 0.0111, "final_en_200.csv"),
    ("if_sara H-1 comb minus control (unpaired, conservative)", 3.7865, 0.0262, 3.7678, 0.0260, "italian_200.csv"),
    ("if_sara H-1 control minus 50k recipe (unpaired, conservative)", 3.7678, 0.0260, 3.7524, 0.0267, "italian_200.csv"),
]


def main():
    out = []
    for label, n, d, se, wins, src in ROWS:
        z = d / se
        out.append([label, n, d, se, round(z, 2), fmt_p(p_normal(z)),
                    "" if wins is None else f"{wins}/{n}", "" if wins is None else fmt_p(sign_test(wins, n)), src])
    for label, ma, ca, mb, cb, src in UNPAIRED:
        d = ma - mb
        se = math.hypot(ca / 1.96, cb / 1.96)
        z = d / se
        out.append([label, 200, round(d, 4), round(se, 4), round(z, 2), fmt_p(p_normal(z)), "", "", src])
    # One listener, 4 sentences, s8 preferred on 3 (listening.csv): one-sided binomial.
    k, n = 3, 4
    p_one = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    out.append(["blind A/B s8 preferred over s7 (one-sided binomial)", n, "", "", "", f"{p_one:.4g}", f"{k}/{n}", "", "listening.csv"])
    with open(os.path.join(HERE, "paired_tests.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["comparison", "n", "mean_diff", "se", "z", "p_two_sided", "wins", "sign_test_p", "source_csv"])
        w.writerows(out)
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
