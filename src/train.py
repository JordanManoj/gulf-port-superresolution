"""Fine-tune an RRDBNet (ESRGAN generator) on WorldStrat pairs.

Modes (set in the YAML config):
  l1  : L1 pixel loss only                     -> fidelity-oriented ("PSNR model")
  gan : L1 + VGG perceptual + adversarial loss -> Real-ESRGAN recipe ("perceptual model")
Init:  realesrgan | satlas | scratch  (the build-vs-reuse ablation)

Usage:  python src/train.py configs/ft_l1_realesrgan.yaml
"""
import copy
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.utils import spectral_norm

sys.path.insert(0, os.path.dirname(__file__))
from srlib import ROOT, SCALE, WEIGHTS, RRDBNet, lpips_dist, psnr, sr_tiled  # noqa: E402


# ---------------------------------------------------------------- discriminator (Real-ESRGAN)
class UNetDiscriminatorSN(nn.Module):
    """U-Net discriminator with spectral norm (Wang et al. 2021), gives per-pixel real/fake logits."""

    def __init__(self, in_ch=3, nf=64):
        super().__init__()
        self.conv0 = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.conv1 = spectral_norm(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False))
        self.conv4 = spectral_norm(nn.Conv2d(nf * 8, nf * 4, 3, 1, 1, bias=False))
        self.conv5 = spectral_norm(nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=False))
        self.conv6 = spectral_norm(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=False))
        self.conv7 = spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=False))
        self.conv8 = spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(nf, 1, 3, 1, 1)

    def forward(self, x):
        lr = lambda t: F.leaky_relu(t, 0.2, inplace=True)
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        x0 = lr(self.conv0(x))
        x1 = lr(self.conv1(x0))
        x2 = lr(self.conv2(x1))
        x3 = lr(self.conv3(x2))
        x4 = lr(self.conv4(up(x3))) + x2
        x5 = lr(self.conv5(up(x4))) + x1
        x6 = lr(self.conv6(up(x5))) + x0
        return self.conv9(lr(self.conv8(lr(self.conv7(x6)))))


# ---------------------------------------------------------------- perceptual loss (VGG19)
class PerceptualLoss(nn.Module):
    """L1 between VGG19 feature maps (pre-activation conv1_2..conv5_4), as in ESRGAN/Real-ESRGAN."""

    LAYERS = {2: 0.1, 7: 0.1, 16: 1.0, 25: 1.0, 34: 1.0}  # conv1_2, conv2_2, conv3_4, conv4_4, conv5_4

    def __init__(self):
        super().__init__()
        import torchvision
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features[:35]
        self.vgg = vgg.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x, y = (x - self.mean) / self.std, (y - self.mean) / self.std
        loss = 0
        for i, layer in enumerate(self.vgg):
            x, y = layer(x), layer(y)
            if i in self.LAYERS:
                loss = loss + self.LAYERS[i] * F.l1_loss(x, y)
        return loss


# ---------------------------------------------------------------- data
class PairSampler:
    """Random aligned crops (LR 32x32 -> HR 128x128) from cached npz pairs, fully inside the valid mask."""

    def __init__(self, paths, lr_patch=32):
        self.items = [np.load(os.path.join(ROOT, p)) for p in paths]
        self.items = [(d["lr"], d["hr"], d["valid"]) for d in self.items]
        self.p = lr_patch

    def sample(self, bs):
        lrs, hrs = [], []
        while len(lrs) < bs:
            lr, hr, valid = random.choice(self.items)
            h, w = lr.shape[:2]
            y, x = random.randint(0, h - self.p), random.randint(0, w - self.p)
            if valid[y * SCALE:(y + self.p) * SCALE, x * SCALE:(x + self.p) * SCALE].mean() < 0.999:
                continue
            a = lr[y:y + self.p, x:x + self.p]
            b = hr[y * SCALE:(y + self.p) * SCALE, x * SCALE:(x + self.p) * SCALE]
            k = random.randint(0, 3)  # dihedral augmentation (satellite images have no "up")
            a, b = np.rot90(a, k), np.rot90(b, k)
            if random.random() < 0.5:
                a, b = a[:, ::-1], b[:, ::-1]
            lrs.append(np.ascontiguousarray(a))
            hrs.append(np.ascontiguousarray(b))
        t = lambda arr: torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).float().div(255)
        return t(lrs), t(hrs)


def evaluate(net, val_paths, device):
    net.eval()
    ps, lp = [], []
    for p in val_paths:
        d = np.load(os.path.join(ROOT, p))
        sr = sr_tiled(net, d["lr"], device=device)
        ps.append(psnr(sr, d["hr"], d["valid"]))
        lp.append(lpips_dist(sr, d["hr"], d["valid"], device=device))
    net.train()
    return float(np.mean(ps)), float(np.mean(lp))


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    seed = cfg.get("seed", 0)
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    dev = "cuda"
    name = cfg["name"]
    out_dir = os.path.join(ROOT, "results", "training", name)
    os.makedirs(out_dir, exist_ok=True)

    pairs = pd.read_csv(os.path.join(ROOT, "data", "pairs", "pairs.csv"))
    train = PairSampler(pairs[pairs.set == "train"].path.tolist(), cfg.get("lr_patch", 32))
    val_paths = pairs[pairs.set == "val"].path.tolist()[: cfg.get("n_val", 30)]

    G = RRDBNet().to(dev)
    if cfg["init"] != "scratch":
        sd = torch.load(os.path.join(ROOT, WEIGHTS[cfg["init"]]), map_location="cpu", weights_only=False)
        G.load_state_dict(sd.get("params_ema", sd.get("params", sd)))
    optG = torch.optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(0.9, 0.99))
    # cosine decay to 10 % of the start LR: big steps early, small steps at the end (val PSNR swung +-4 dB with a constant LR)
    schedG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, cfg["iters"], eta_min=cfg["lr_g"] * 0.1)
    # EMA of generator weights (Real-ESRGAN uses 0.999): smooths step-to-step noise; we validate and save the EMA model
    ema = copy.deepcopy(G).eval().requires_grad_(False)
    ema_decay = cfg.get("ema", 0.999)
    gan = cfg["mode"] == "gan"
    if gan:
        D = UNetDiscriminatorSN().to(dev)
        sd = torch.load(os.path.join(ROOT, cfg["init_d"]), map_location="cpu", weights_only=False)["params"]
        D.load_state_dict(sd)
        optD = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.9, 0.99))
        percep = PerceptualLoss().to(dev)
        bce = nn.BCEWithLogitsLoss()
    log, best = [], None
    v_psnr, v_lpips = evaluate(ema, val_paths, dev)
    log.append(dict(it=0, val_psnr=v_psnr, val_lpips=v_lpips))
    print(f"[{name}] it 0 val PSNR {v_psnr:.2f} LPIPS {v_lpips:.3f}", flush=True)
    t0 = time.time()
    for it in range(1, cfg["iters"] + 1):
        lr_b, hr_b = train.sample(cfg["batch"])
        lr_b, hr_b = lr_b.to(dev), hr_b.to(dev)
        # ---- generator step
        # fp32 on purpose: fp16 autocast gave NaN losses at ~50 its on the GTX 1650 (no tensor cores, little speed-up)
        sr = G(lr_b)
        l_pix = F.l1_loss(sr, hr_b)
        loss = cfg.get("w_pix", 1.0) * l_pix
        if gan:
            for p in D.parameters():
                p.requires_grad_(False)
            l_per = percep(sr, hr_b)
            l_gan = bce(D(sr), torch.ones_like(sr[:, :1]))
            loss = loss + cfg["w_per"] * l_per + cfg["w_gan"] * l_gan
        optG.zero_grad(set_to_none=True)
        loss.backward()
        optG.step()
        schedG.step()
        with torch.no_grad():
            for pe, pg in zip(ema.parameters(), G.parameters()):
                pe.mul_(ema_decay).add_(pg.detach(), alpha=1 - ema_decay)
        # ---- discriminator step
        if gan:
            for p in D.parameters():
                p.requires_grad_(True)
            l_d = bce(D(hr_b), torch.ones_like(hr_b[:, :1])) + bce(D(sr.detach()), torch.zeros_like(hr_b[:, :1]))
            optD.zero_grad(set_to_none=True)
            l_d.backward()
            optD.step()
        if it % cfg.get("log_every", 50) == 0:
            msg = f"[{name}] it {it} l_pix {l_pix.item():.4f}"
            if gan:
                msg += f" l_per {l_per.item():.3f} l_gan {l_gan.item():.3f} l_d {l_d.item():.3f}"
            print(msg + f" ({(time.time() - t0) / it:.2f}s/it)", flush=True)
        if it % cfg["val_every"] == 0 or it == cfg["iters"]:
            v_psnr, v_lpips = evaluate(ema, val_paths, dev)
            log.append(dict(it=it, val_psnr=v_psnr, val_lpips=v_lpips, l_pix=l_pix.item()))
            print(f"[{name}] it {it} val PSNR {v_psnr:.2f} LPIPS {v_lpips:.3f}", flush=True)
            # model selection: PSNR for the L1 model, LPIPS for the perceptual model
            score = -v_lpips if gan else v_psnr
            if best is None or score > best:
                best = score
                torch.save({"params": ema.state_dict(), "it": it}, os.path.join(ROOT, cfg["save_to"]))
            json.dump(log, open(os.path.join(out_dir, "log.json"), "w"), indent=1)
    print(f"[{name}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
