"""Download a small, targeted subset of WorldStrat v1.1 (Zenodo) via HTTP range requests.

The full dataset is ~104 GB. We only need a few hundred tiles, so we:
  1. read each zip's central directory once (cached to disk),
  2. pick tiles from the official stratified split (+ every Gulf tile, held out),
  3. fetch only the needed members (HR RGB png + 1-3 Sentinel-2 L2A revisits) in parallel.

Usage:  python src/download_worldstrat.py --n-train 600 --n-val 60 --n-test 120
"""
import argparse
import os
import pickle
import struct
import time
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
from remote_zip import open_remote_zip  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(ROOT, "data", "worldstrat")
BASE = "https://zenodo.org/records/15382551/files/{}.zip?download=1"
GULF_BOX = dict(lat=(22.0, 30.0), lon=(47.0, 58.0))  # Arabian/Persian Gulf region


def get_index(name):
    """Map member filename -> (header_offset, compress_size, compress_type). Cached."""
    cache = os.path.join(WS, f"_index_{name}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    z = open_remote_zip(BASE.format(name))
    idx = {i.filename: (i.header_offset, i.compress_size, i.compress_type) for i in z.infolist()}
    pickle.dump(idx, open(cache, "wb"))
    return idx


def fetch_member(sess, name, entry, out_path):
    """Fetch one zip member with a single Range request and inflate it."""
    if os.path.exists(out_path):
        return out_path
    off, csize, ctype = entry
    url = BASE.format(name)
    # local header = 30 bytes + filename + extra; extra field length unknown -> over-fetch 1 KB
    end = off + 30 + 1024 + csize
    for attempt in range(5):
        try:
            r = sess.get(url, headers={"Range": f"bytes={off}-{end}"}, timeout=180)
            if r.status_code == 206:
                break
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)  # back off: Zenodo throttles bursts of range requests
    else:
        raise IOError(f"failed {out_path}")
    buf = r.content
    assert buf[:4] == b"PK\x03\x04", "bad local header"
    nlen, xlen = struct.unpack("<HH", buf[26:30])
    data = buf[30 + nlen + xlen: 30 + nlen + xlen + csize]
    if ctype == 8:
        data = zlib.decompress(data, -15)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".part"
    open(tmp, "wb").write(data)
    os.replace(tmp, out_path)
    return out_path


def select_tiles(n_train, n_val, n_test, seed=42):
    meta = pd.read_csv(os.path.join(WS, "metadata.csv"))
    split = pd.read_csv(os.path.join(WS, "stratified_train_val_test_split.csv"))
    tiles = meta.groupby("tile").first().reset_index()[["tile", "lat", "lon", "IPCC Class", "SMOD Class"]]
    tiles = tiles.merge(split[["tile", "split"]], on="tile")
    gulf = tiles.lat.between(*GULF_BOX["lat"]) & tiles.lon.between(*GULF_BOX["lon"])
    tiles["set"] = None
    tiles.loc[gulf, "set"] = "gulf_test"
    rest = tiles[~gulf]

    def pick(df, n):
        # oversample built-up areas: half Settlement, half everything else
        s = df[df["IPCC Class"] == "Settlement"]
        o = df[df["IPCC Class"] != "Settlement"]
        k = min(len(s), n // 2)
        return pd.concat([s.sample(k, random_state=seed), o.sample(min(len(o), n - k), random_state=seed)])

    for sp, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        chosen = pick(rest[rest.split == sp], n)
        tiles.loc[chosen.index, "set"] = sp
    sel = tiles[tiles.set.notna()].copy()

    # choose LR revisits: low cloud, closest in time to the HR acquisition
    meta["absdelta"] = meta.delta.abs()
    good = meta[meta.cloud_cover <= 0.05].sort_values(["tile", "absdelta", "cloud_cover"])
    revs = good.groupby("tile").n.apply(list).to_dict()
    fallback = meta.sort_values(["tile", "cloud_cover", "absdelta"]).groupby("tile").n.apply(list).to_dict()
    sel["revisits"] = [
        (revs.get(t) or fallback[t])[: (3 if s == "train" else 1)] for t, s in zip(sel.tile, sel.set)
    ]
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=120)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    sel = select_tiles(a.n_train, a.n_val, a.n_test)
    print(sel.set.value_counts().to_string())
    hr_idx, lr_idx = get_index("hr_dataset"), get_index("lr_dataset_l2a")

    jobs = []
    # evaluation sets first, so a throttled/partial download still yields complete test sets
    prio = {"gulf_test": 0, "test": 1, "val": 2, "train": 3}
    sel = sel.sort_values("set", key=lambda c: c.map(prio), kind="stable")
    for t, revs in zip(sel.tile, sel.revisits):
        jobs.append(("hr_dataset", hr_idx[f"hr_dataset/{t}/{t}_rgb.png"], os.path.join(WS, "hr", f"{t}_rgb.png")))
        for n in revs:
            m = f"lr_dataset/{t}/L2A/{t}-{n}-L2A_data.tiff"
            jobs.append(("lr_dataset_l2a", lr_idx[m], os.path.join(WS, "lr", t, f"{t}-{n}-L2A_data.tiff")))
    print(f"{len(jobs)} files to fetch")

    sess = requests.Session()
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch_member, sess, *j) for j in jobs]
        failed = 0
        for f in as_completed(futs):
            try:
                f.result()
            except IOError as e:
                failed += 1
                print("  skip:", e, flush=True)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    print(f"failed files: {failed}")
    sel = sel.sort_index()
    sel.to_csv(os.path.join(WS, "subset.csv"), index=False)
    print("done ->", os.path.join(WS, "subset.csv"))


if __name__ == "__main__":
    main()
