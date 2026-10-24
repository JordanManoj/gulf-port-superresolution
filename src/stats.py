"""Paired bootstrap confidence intervals: are the differences between methods real or noise?

Resamples tiles with replacement (10,000 times) and reports 95 % CIs for
  - mean PSNR / LPIPS / hallucination difference of each method vs bicubic (paired by tile)
  - micro-averaged detection F1 of each method
Usage:  python src/stats.py
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = 10000


def main():
    df = pd.read_csv(os.path.join(ROOT, "results", "metrics", "per_tile.csv"))
    rng = np.random.default_rng(0)
    rows = []
    for s, d in df.groupby("set"):
        tiles = d.tile.unique()
        idx = rng.integers(0, len(tiles), size=(B, len(tiles)))
        for m in d.method.unique():
            r = dict(set=s, method=m, n_tiles=len(tiles))
            for col in ["psnr", "lpips", "hallucination"]:
                piv = d.pivot(index="tile", columns="method", values=col).loc[tiles]
                diff = (piv[m] - piv["bicubic"]).values
                boots = diff[idx].mean(1)
                r[f"d_{col}"] = diff.mean()
                r[f"d_{col}_lo"], r[f"d_{col}_hi"] = np.percentile(boots, [2.5, 97.5])
            dm = d[d.method == m].set_index("tile").loc[tiles]
            tp, fp, fn = dm.det_tp.values[idx].sum(1), dm.det_fp.values[idx].sum(1), dm.det_fn.values[idx].sum(1)
            f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
            r["det_f1"] = 2 * dm.det_tp.sum() / max(2 * dm.det_tp.sum() + dm.det_fp.sum() + dm.det_fn.sum(), 1)
            r["det_f1_lo"], r["det_f1_hi"] = np.percentile(f1, [2.5, 97.5])
            r["det_ref_objects"] = int(dm.det_ref.sum())
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "results", "metrics", "bootstrap_ci.csv"), index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
