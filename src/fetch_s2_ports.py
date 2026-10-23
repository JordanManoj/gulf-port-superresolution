"""Fetch cloud-free Sentinel-2 L2A RGB crops over the UAE port AOIs (aoi/*.geojson).

Source: Element 84 Earth Search STAC (AWS Open Data, no account needed) -> Cloud-Optimised GeoTIFFs,
so only the port window is downloaded. Native 10 m UTM grid is kept (no resampling).
Several dates from the SAME MGRS tile are fetched so they are pixel-aligned (used for the
reference-free temporal-consistency check).

Usage:  python src/fetch_s2_ports.py --n-dates 4
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(__file__))
from srlib import ROOT, refl_to_u8  # noqa: E402

STAC = "https://earth-search.aws.element84.com/v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dates", type=int, default=4)
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-10-20")
    a = ap.parse_args()
    client = Client.open(STAC)
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
    manifest = {}
    for gj in sorted(glob.glob(os.path.join(ROOT, "aoi", "*.geojson"))):
        feat = json.load(open(gj))["features"][0]
        name = feat["properties"]["name"]
        ring = np.array(feat["geometry"]["coordinates"][0])
        bbox = [ring[:, 0].min(), ring[:, 1].min(), ring[:, 0].max(), ring[:, 1].max()]
        items = list(client.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{a.start}/{a.end}",
                                   query={"eo:cloud_cover": {"lt": 5}}, max_items=200).items())
        # keep the MGRS tile that fully contains the AOI and has most scenes
        by_tile = {}
        for it in items:
            by_tile.setdefault(it.properties["grid:code"], []).append(it)
        out_dir = os.path.join(ROOT, "data", "s2_ports", name)
        os.makedirs(out_dir, exist_ok=True)
        chosen = []
        for tile, its in sorted(by_tile.items(), key=lambda kv: -len(kv[1])):
            its = sorted(its, key=lambda i: i.properties["eo:cloud_cover"])
            for it in its:
                if len(chosen) >= a.n_dates:
                    break
                if any(c["date"] == str(it.datetime.date()) for c in chosen):
                    continue
                bands, prof = [], None
                for b in ["red", "green", "blue"]:
                    asset = it.assets[b]
                    rb = asset.extra_fields["raster:bands"][0]
                    with rasterio.open(asset.href) as src:
                        bb = transform_bounds("EPSG:4326", src.crs, *bbox)
                        win = from_bounds(*bb, transform=src.transform).round_offsets().round_lengths()
                        arr = src.read(1, window=win, boundless=False)
                        if prof is None:
                            prof = dict(driver="GTiff", crs=src.crs, transform=src.window_transform(win),
                                        width=arr.shape[1], height=arr.shape[0], count=3, dtype="float32")
                    if (arr == 0).mean() > 0.01:  # AOI not fully inside this tile / nodata
                        bands = None
                        break
                    bands.append(arr.astype(np.float32) * rb.get("scale", 1e-4) + rb.get("offset", 0.0))
                if bands is None:
                    continue
                refl = np.stack(bands)
                date = str(it.datetime.date())
                with rasterio.open(os.path.join(out_dir, f"{date}_refl.tif"), "w", **prof) as dst:
                    dst.write(refl)
                import cv2
                cv2.imwrite(os.path.join(out_dir, f"{date}_rgb.png"),
                            cv2.cvtColor(refl_to_u8(refl.transpose(1, 2, 0)), cv2.COLOR_RGB2BGR))
                chosen.append(dict(date=date, id=it.id, cloud=it.properties["eo:cloud_cover"], mgrs=tile,
                                   shape=list(refl.shape)))
                print(name, date, it.id, refl.shape, flush=True)
            if chosen:
                break
        manifest[name] = chosen
    json.dump(manifest, open(os.path.join(ROOT, "data", "s2_ports", "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
