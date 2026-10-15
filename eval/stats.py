"""Summary statistics of per-row scores, as the paper reports them."""
import math

import numpy as np
from scipy import stats


def ci95(x):
    """Half-width of the 95% interval of a mean as the paper's tables give it: 1.96 x population SD / sqrt(n)."""
    x = np.asarray(x, dtype=np.float64)
    return 1.96 * float(x.std()) / math.sqrt(len(x))


def paired(b, a):
    """b - a on the same rows: mean difference, its standard error (sample SD / sqrt(n)), the paired t-test, the
    rows on which b is higher and the exact sign test (rows with equal scores left out). p_normal is the normal
    approximation used for the paper's significance values (paper/data/stats.py)."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    n = len(d)
    if n < 2:
        raise ValueError("a paired comparison needs at least two rows")
    mean, se = float(d.mean()), float(d.std(ddof=1)) / math.sqrt(n)
    t = mean / se if se > 0 else (0.0 if mean == 0 else math.copysign(math.inf, mean))
    better, worse = int((d > 0).sum()), int((d < 0).sum())
    sign = stats.binomtest(better, better + worse).pvalue if better + worse else 1.0
    return {"n": n, "mean_diff": round(mean, 4), "se": round(se, 4), "t": round(t, 2),
            "p": float(f"{2 * stats.t.sf(abs(t), n - 1):.3g}"), "p_normal": float(f"{2 * stats.norm.sf(abs(t)):.3g}"),
            "better": better, "worse": worse, "sign_test_p": float(f"{sign:.3g}")}
