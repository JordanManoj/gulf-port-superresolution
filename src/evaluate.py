"""Evaluate all SR methods on the WorldStrat test sets (global test + held-out Gulf test).

For every tile and method:
  image quality : PSNR, SSIM, LPIPS (vs colour-matched, registered HR at 2.5 m)
  structure     : edge F1 (Canny, 2 px tolerance)
  correctness   : improvement / omission / hallucination patch rates (OpenSR-test style)
                  + "invented detail" rate (SR has texture where HR is flat)
  consistency   : PSNR of SR downsampled back to 10 m vs the LR input
  usefulness    : YOLO11m-OBB (DOTA) detections on SR vs detections on real HR (F1 @ IoU 0.3)

SR outputs are colour-harmonised to the bicubic image before scoring (as OpenSR-test does),
so a model is not punished for NAIP-style colours, only for wrong geometry.

Usage:  python src/evaluate.py --methods bicubic realesrgan satlas ft_l1 ft_gan
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from srlib import (ROOT, WEIGHTS, bicubic, correctness, edge_f1, hallucination_maps, hist_match,  # noqa: E402
                   load_model, lpips_dist, lr_consistency, psnr, sr_tiled, ssim)

DET_CONF = 0.25
DET_IMGSZ = 1024
SAT_TILE = 32  # Satlas was trained on 32x32 chunks; we give it the same context


def run_detector(det, img):
    r = det.predict(img[..., ::-1], imgsz=DET_IMGSZ, conf=DET_CONF, verbose=False)[0]
    if r.obb is None or len(r.obb) == 0:
        return np.zeros((0, 4)), np.zeros(0, int)
    return r.obb.xyxy.cpu().numpy(), r.obb.cls.cpu().numpy().astype(int)


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(pred, pred_cls, ref, ref_cls, thr=0.3):
    """Greedy one-to-one matching (class-aware). Returns TP, FP, FN."""
    iou = box_iou(pred, ref)
    iou[pred_cls[:, None] != ref_cls[None, :]] = 0
    tp, used = 0, set()
    for i in np.argsort(-iou.max(1) if len(ref) else np.zeros(len(pred))):
        if len(ref) == 0:
            break
        j = int(np.argmax(iou[i]))
        if iou[i, j] >= thr and j not in used:
            used.add(j)
            tp += 1
    return tp, len(pred) - tp, len(ref) - tp


def in_mask(boxes, mask):
    if len(boxes) == 0:
        return np.zeros(0, bool)
    cx = ((boxes[:, 0] + boxes[:, 2]) / 2).astype(int).clip(0, mask.shape[1] - 1)
    cy = ((boxes[:, 1] + boxes[:, 3]) / 2).astype(int).clip(0, mask.shape[0] - 1)
    return mask[cy, cx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["bicubic", "realesrgan", "satlas", "ft_l1", "ft_gan"])
    ap.add_argument("--sets", nargs="+", default=["test", "gulf_test"])
    ap.add_argument("--save-sr", action="store_true", help="save SR pngs for figures/demo")
    ap.add_argument("--out", default="results/metrics/per_tile.csv")
    a = ap.parse_args()

    from ultralytics import YOLO
    det = YOLO(os.path.join(ROOT, "weights", "yolo11m-obb.pt"))
    pairs = pd.read_csv(os.path.join(ROOT, "data", "pairs", "pairs.csv"))
    pairs = pairs[pairs.set.isin(a.sets)].drop_duplicates("tile")  # one revisit per tile
    models = {m: load_model(m) for m in a.methods if m != "bicubic" and os.path.exists(os.path.join(ROOT, WEIGHTS.get(m, "x")))}
    rows = []
    for k, (_, p) in enumerate(pairs.iterrows()):
        d = np.load(os.path.join(ROOT, p.path))
        lr, hr, valid = d["lr"], d["hr"], d["valid"]
        base = bicubic(lr)
        ref_boxes, ref_cls = run_detector(det, hr)
        keep = in_mask(ref_boxes, valid)
        ref_boxes, ref_cls = ref_boxes[keep], ref_cls[keep]
        for m in a.methods:
            if m == "bicubic":
                sr_raw = base
            elif m in models:
                sr_raw = sr_tiled(models[m], lr, tile=SAT_TILE if m == "satlas" else 64,
                                  pad=4 if m == "satlas" else 8)
            else:
                continue
            sr = hist_match(sr_raw, base, valid) if m != "bicubic" else base
            c = correctness(sr, base, hr, valid)
            h = hallucination_maps(sr, hr, valid)
            ef, ep, er = edge_f1(sr, hr, valid)
            pb, pc = run_detector(det, sr)
            keep = in_mask(pb, valid)
            tp, fp, fn = match(pb[keep], pc[keep], ref_boxes, ref_cls)
            rows.append(dict(set=p.set, tile=p.tile, method=m,
                             psnr=psnr(sr, hr, valid), ssim=ssim(sr, hr, valid),
                             lpips=lpips_dist(sr, hr, valid), edge_f1=ef,
                             improvement=c["improvement"], omission=c["omission"],
                             hallucination=c["hallucination"], invented_detail=h["hall_rate"],
                             missed_detail=h["miss_rate"], lr_consistency=lr_consistency(sr_raw, lr),
                             det_tp=tp, det_fp=fp, det_fn=fn, det_ref=len(ref_boxes)))
            if a.save_sr:
                o = os.path.join(ROOT, "results", "sr_outputs", p.set, p.tile)
                os.makedirs(o, exist_ok=True)
                cv2.imwrite(os.path.join(o, f"{m}.png"), cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))
                if m == a.methods[0]:
                    cv2.imwrite(os.path.join(o, "hr.png"), cv2.cvtColor(hr, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(o, "lr.png"), cv2.cvtColor(lr, cv2.COLOR_RGB2BGR))
                    np.save(os.path.join(o, "valid.npy"), valid)
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(pairs)} tiles", flush=True)
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    # summary: mean of per-tile metrics; detection F1 is micro-averaged over all tiles
    agg = df.groupby(["set", "method"]).agg(
        n=("tile", "size"), psnr=("psnr", "mean"), ssim=("ssim", "mean"), lpips=("lpips", "mean"),
        edge_f1=("edge_f1", "mean"), improvement=("improvement", "mean"), omission=("omission", "mean"),
        hallucination=("hallucination", "mean"), invented_detail=("invented_detail", "mean"),
        lr_consistency=("lr_consistency", "mean"), tp=("det_tp", "sum"), fp=("det_fp", "sum"),
        fn=("det_fn", "sum"))
    agg["det_precision"] = agg.tp / (agg.tp + agg.fp).clip(lower=1)
    agg["det_recall"] = agg.tp / (agg.tp + agg.fn).clip(lower=1)
    agg["det_f1"] = 2 * agg.tp / (2 * agg.tp + agg.fp + agg.fn).clip(lower=1)
    agg = agg.reset_index()
    agg.to_csv(out.replace(".csv", "_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(agg.round(4).to_string())


if __name__ == "__main__":
    main()
