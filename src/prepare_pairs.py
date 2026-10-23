"""Turn the downloaded WorldStrat subset into aligned, cached (LR, HR, mask) pairs.

Each LR revisit becomes one pair: HR is resampled to 2.5 m, colour-matched to that revisit
and sub-pixel registered (see srlib.prepare_pair). Results -> data/pairs/<set>/<tile>__<n>.npz
plus data/pairs/pairs.csv with registration diagnostics.

Usage:  python src/prepare_pairs.py
"""
import ast
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from srlib import ROOT, prepare_pair  # noqa: E402

WS = os.path.join(ROOT, "data", "worldstrat")
OUT = os.path.join(ROOT, "data", "pairs")


def main():
    sub = pd.read_csv(os.path.join(WS, "subset.csv"))
    rows = []
    for t, s, revs in zip(sub.tile, sub.set, sub.revisits):
        hr_path = os.path.join(WS, "hr", f"{t}_rgb.png")
        for n in ast.literal_eval(revs):
            lr_path = os.path.join(WS, "lr", t, f"{t}-{n}-L2A_data.tiff")
            out = os.path.join(OUT, s, f"{t}__{n}.npz")
            if not (os.path.exists(hr_path) and os.path.exists(lr_path)):
                continue
            if not os.path.exists(out):
                try:
                    lr, hr, valid, (dx, dy, resp) = prepare_pair(lr_path, hr_path)
                except Exception as e:  # corrupt / empty tile
                    print("skip", t, n, e)
                    continue
                os.makedirs(os.path.dirname(out), exist_ok=True)
                np.savez_compressed(out, lr=lr, hr=hr, valid=valid, shift=np.array([dx, dy, resp]))
            d = np.load(out)
            rows.append(dict(set=s, tile=t, n=n, path=os.path.relpath(out, ROOT),
                             dx=d["shift"][0], dy=d["shift"][1], pc_resp=d["shift"][2],
                             valid_frac=d["valid"].mean(), lr_mean=d["lr"].mean(),
                             lr_sat=(d["lr"] == 255).mean()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "pairs.csv"), index=False)
    print(df.groupby("set").agg(n=("tile", "size"), tiles=("tile", "nunique"),
                                shift_px=("dx", lambda x: np.abs(x).mean()), valid=("valid_frac", "mean"),
                                sat=("lr_sat", "mean")).to_string())


if __name__ == "__main__":
    main()
