"""Make every figure used in the README and slides from the CSVs/PNGs written by the eval scripts.

Usage:  python src/make_figures.py
"""
import glob
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from srlib import ROOT, correctness, hallucination_maps  # noqa: E402

FIG = os.path.join(ROOT, "results", "figures")
MET = os.path.join(ROOT, "results", "metrics")
SR = os.path.join(ROOT, "results", "sr_outputs")
os.makedirs(FIG, exist_ok=True)

LABEL = {"bicubic": "Bicubic", "realesrgan": "Real-ESRGAN\n(pretrained)", "satlas": "Satlas\n(pretrained)",
         "ft_l1": "Fine-tuned\n(L1)", "ft_gan": "Fine-tuned\n(GAN)"}
COLOR = {"bicubic": "#8c8c8c", "realesrgan": "#4C72B0", "satlas": "#DD8452", "ft_l1": "#55A868", "ft_gan": "#C44E52"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def metric_bars(summ):
    methods = [m for m in LABEL if m in summ.method.unique()]
    panels = [("psnr", "PSNR (dB) ↑"), ("ssim", "SSIM ↑"), ("lpips", "LPIPS ↓"), ("edge_f1", "Edge F1 ↑"),
              ("hallucination", "Hallucinated patches ↓"), ("det_f1", "Detection F1 vs HR ↑")]
    for s in summ.set.unique():
        d = summ[summ.set == s].set_index("method").loc[methods]
        fig, axs = plt.subplots(1, len(panels), figsize=(17, 3.4))
        for ax, (col, title) in zip(axs, panels):
            vals = d[col].values
            ax.bar(range(len(methods)), vals, color=[COLOR[m] for m in methods])
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([LABEL[m] for m in methods], fontsize=7)
            ax.set_title(title, fontsize=10)
            lo, hi = vals.min(), vals.max()
            if col == "psnr":
                ax.set_ylim(lo - 1.5, hi + 0.8)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.3f}" if v < 2 else f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        name = "WorldStrat global test (120 tiles)" if s == "test" else "Held-out Gulf tiles (13 tiles, never trained on)"
        fig.suptitle(name, fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"metrics_{s}.png"), bbox_inches="tight")
        plt.close(fig)


def perception_distortion(summ):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, s in zip(axs, ["test", "gulf_test"]):
        d = summ[summ.set == s]
        for _, r in d.iterrows():
            ax.scatter(r.psnr, r.lpips, s=60 + 1500 * r.hallucination, color=COLOR[r.method], alpha=.85,
                       edgecolor="k", lw=.5)
            ax.annotate(LABEL[r.method].replace("\n", " "), (r.psnr, r.lpips), fontsize=7, xytext=(5, 4),
                        textcoords="offset points")
        ax.set_xlabel("PSNR (dB) →  less distortion")
        ax.set_ylabel("LPIPS ↓  more perceptual")
        ax.set_title("Global test" if s == "test" else "Gulf tiles")
    fig.suptitle("Perception–distortion plane (marker size = hallucinated-patch rate)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "perception_distortion.png"), bbox_inches="tight")
    plt.close(fig)


def ablation_curves():
    logs = {}
    for f in glob.glob(os.path.join(ROOT, "results", "training", "*", "log.json")):
        logs[os.path.basename(os.path.dirname(f))] = pd.DataFrame(json.load(open(f)))
    abl = {k: v for k, v in logs.items() if k.startswith("ablation")}
    if not abl:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    names = {"ablation_l1_realesrgan": ("init: Real-ESRGAN", "#4C72B0"), "ablation_l1_satlas": ("init: Satlas", "#DD8452"),
             "ablation_l1_scratch": ("init: random (scratch)", "#333333")}
    for k, (lab, c) in names.items():
        if k in abl:
            axs[0].plot(abl[k].it, abl[k].val_psnr, "o-", label=lab, color=c)
            axs[1].plot(abl[k].it, abl[k].val_lpips, "o-", label=lab, color=c)
    axs[0].set(xlabel="iteration", ylabel="val PSNR (dB) ↑", title="Same data, loss and budget; only the init differs")
    axs[1].set(xlabel="iteration", ylabel="val LPIPS ↓")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ablation_init.png"), bbox_inches="tight")
    plt.close(fig)
    if "ft_gan" in logs:
        g = logs["ft_gan"]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        ax[0].plot(g.it, g.val_psnr, "o-", color="#C44E52")
        ax[0].set(xlabel="iteration", ylabel="val PSNR ↑", title="GAN fine-tune: PSNR falls…")
        ax[1].plot(g.it, g.val_lpips, "o-", color="#C44E52")
        ax[1].set(xlabel="iteration", ylabel="val LPIPS ↓", title="…while LPIPS improves")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "gan_training.png"), bbox_inches="tight")
        plt.close(fig)


def comparison_grid(per_tile, tiles, fname, title, crop=192):
    methods = [m for m in LABEL if m in per_tile.method.unique()]
    cols = ["lr"] + methods + ["hr"]
    fig, axs = plt.subplots(len(tiles), len(cols), figsize=(2.1 * len(cols), 2.2 * len(tiles)))
    axs = np.atleast_2d(axs)
    for i, (s, t) in enumerate(tiles):
        d = os.path.join(SR, s, t)
        hr = rgb(os.path.join(d, "hr.png"))
        valid = np.load(os.path.join(d, "valid.npy"))
        # crop where the HR has most structure (inside the valid area)
        e = cv2.blur(np.abs(cv2.Laplacian(cv2.cvtColor(hr, cv2.COLOR_RGB2GRAY), cv2.CV_32F)), (crop, crop)) * valid
        y, x = np.unravel_index(np.argmax(e), e.shape)
        y0 = int(np.clip(y - crop // 2, 0, hr.shape[0] - crop)); x0 = int(np.clip(x - crop // 2, 0, hr.shape[1] - crop))
        for j, c in enumerate(cols):
            if c == "lr":
                im = cv2.resize(rgb(os.path.join(d, "lr.png")), (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                im = rgb(os.path.join(d, f"{c}.png"))
            axs[i, j].imshow(im[y0:y0 + crop, x0:x0 + crop])
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            if i == 0:
                axs[i, j].set_title({"lr": "Sentinel-2\n10 m (input)", "hr": "SPOT reference\n2.5 m"}.get(c, LABEL.get(c, c)), fontsize=8)
            if c in methods:
                r = per_tile[(per_tile.tile == t) & (per_tile.method == c)].iloc[0]
                axs[i, j].set_xlabel(f"{r.psnr:.1f} dB | hal {r.hallucination:.0%}", fontsize=7)
        axs[i, 0].set_ylabel(t[:22], fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight")
    plt.close(fig)


def hallucination_figure(per_tile, s, t, method, fname):
    """Difference + correctness map for one tile: where did the model invent structure?"""
    d = os.path.join(SR, s, t)
    hr, base, sr = rgb(os.path.join(d, "hr.png")), rgb(os.path.join(d, "bicubic.png")), rgb(os.path.join(d, f"{method}.png"))
    valid = np.load(os.path.join(d, "valid.npy"))
    c = correctness(sr, base, hr, valid)
    h = hallucination_maps(sr, hr, valid)
    cls = cv2.resize(c["cls"].astype(np.float32), (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = (hr * 0.55).astype(np.uint8).copy()
    overlay[cls == 0] = (0.5 * overlay[cls == 0] + [0, 90, 200]).clip(0, 255)   # improvement: blue
    overlay[cls == 2] = (0.5 * overlay[cls == 2] + [230, 30, 30]).clip(0, 255)  # hallucination: red
    diff = np.abs(sr.astype(np.float32) - hr.astype(np.float32)).mean(-1) * valid
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.8))
    for ax, im, ti in zip(axs, [base, sr, hr], ["Bicubic", LABEL[method].replace("\n", " "), "SPOT reference"]):
        ax.imshow(im); ax.set_title(ti, fontsize=9)
    axs[3].imshow(diff, cmap="magma", vmax=60); axs[3].set_title("|SR − HR| difference", fontsize=9)
    axs[4].imshow(overlay)
    axs[4].contour(h["hall_map"], levels=[0.5], colors="yellow", linewidths=0.4)
    axs[4].set_title("blue = real detail recovered\nred = hallucinated patch, yellow = invented texture", fontsize=8)
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{t}: {method}, hallucination {c['hallucination']:.0%}, improvement {c['improvement']:.0%}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight")
    plt.close(fig)


def port_figures():
    ports = os.path.join(ROOT, "results", "ports")
    if not os.path.isdir(ports):
        return
    det = json.load(open(os.path.join(MET, "ports_detections.json")))
    for port in sorted(os.listdir(ports)):
        dates = sorted(os.listdir(os.path.join(ports, port)))
        date = dates[0]
        methods = [m for m in LABEL if os.path.exists(os.path.join(ports, port, date, f"{m}.png"))]
        ims = {m: rgb(os.path.join(ports, port, date, f"{m}.png")) for m in methods}
        h, w = ims["bicubic"].shape[:2]
        cy, cx, c = h // 2, w // 2, 320
        fig, axs = plt.subplots(1, len(methods), figsize=(3.3 * len(methods), 3.6))
        for ax, m in zip(axs, methods):
            ax.imshow(ims[m][cy - c:cy + c, cx - c:cx + c])
            dd = det[f"{port}/{date}/{m}"]
            for poly, cl in zip(dd["boxes"], dd["classes"]):
                p = np.array(poly) - [cx - c, cy - c]
                if (p.min() >= 0) and (p.max() < 2 * c):
                    ax.add_patch(plt.Polygon(p, fill=False, ec="cyan" if cl == "ship" else "yellow", lw=0.8))
            n = len(dd["classes"])
            ax.set_title(f"{LABEL[m].replace(chr(10), ' ')}\n{dd['classes'].count('ship')} ships, {n} detections (full AOI)", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{port.replace('_', ' ').title()}, Sentinel-2 {date}, x4 (cyan = ship detections)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"port_{port}.png"), bbox_inches="tight")
        plt.close(fig)


def main():
    per_tile = pd.read_csv(os.path.join(MET, "per_tile.csv"))
    summ = pd.read_csv(os.path.join(MET, "per_tile_summary.csv"))
    metric_bars(summ)
    perception_distortion(summ)
    ablation_curves()
    # success cases: where the fine-tuned L1 model gains most over bicubic; failures: highest GAN/Satlas hallucination
    piv = per_tile.pivot_table(index=["set", "tile"], columns="method", values="psnr")
    gain = (piv["ft_l1"] - piv["bicubic"]).sort_values(ascending=False) if "ft_l1" in piv else piv["bicubic"]
    comparison_grid(per_tile, list(gain.index[:3]), "examples_success.png", "Where SR helps most (largest PSNR gain of fine-tuned L1 over bicubic)")
    gulf = [i for i in piv.index if i[0] == "gulf_test"][:4]
    comparison_grid(per_tile, gulf, "examples_gulf.png", "Held-out Gulf tiles")
    hal = per_tile.pivot_table(index=["set", "tile"], columns="method", values="hallucination")
    for m in ["ft_gan", "satlas", "realesrgan"]:
        if m in hal:
            worst = hal[m].sort_values(ascending=False).index[:3]
            comparison_grid(per_tile, list(worst), f"failures_{m}.png", f"Failure cases: highest hallucination for {LABEL[m].replace(chr(10), ' ')}")
            s, t = worst[0]
            hallucination_figure(per_tile, s, t, m, f"hallucination_map_{m}.png")
    if "ft_gan" in hal:
        s, t = gulf[0]
        hallucination_figure(per_tile, s, t, "ft_gan", "hallucination_map_gulf_ft_gan.png")
        hallucination_figure(per_tile, s, t, "ft_l1", "hallucination_map_gulf_ft_l1.png")
    port_figures()
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
