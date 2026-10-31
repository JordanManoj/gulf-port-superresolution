"""Core library: SR models, Sentinel-2 / WorldStrat preprocessing, registration, metrics.

Kept in one module so the whole pipeline is easy to read top-to-bottom.
"""
import os

import cv2
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALE = 4
# WorldStrat L2A tiffs hold 12 bands: B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12 -> RGB = B4,B3,B2
RGB_IDX = [3, 2, 1]
# Linear stretch reflectance -> 8 bit. 0.40 = 99th percentile of Gulf-tile reflectance (2% clipped; 0.30 clipped 12%)
# (Sen2Cor TCI uses 0.25, which clips desert scenes). Chosen from train-set percentiles.
REFL_MAX = 0.40


# --------------------------------------------------------------------------------------
# Model: RRDBNet (ESRGAN generator). Same definition as BasicSR (BSD-3, X. Wang et al.),
# re-implemented here to avoid BasicSR's broken torchvision import on recent versions.
# Both RealESRGAN_x4plus and Satlas esrgan_1S2 load into this exact architecture.
# --------------------------------------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


WEIGHTS = {
    "realesrgan": "weights/RealESRGAN_x4plus.pth",   # generic photos (DF2K/OST), Real-ESRGAN degradations
    "satlas": "weights/esrgan_1S2.pth",              # Sentinel-2 -> NAIP (USA), single image
    "ft_l1": "weights/finetuned_l1.pth",             # ours: fine-tuned on WorldStrat, L1 loss only
    "ft_gan": "weights/finetuned_gan.pth",           # ours: fine-tuned on WorldStrat, L1+perceptual+GAN
}


def load_model(name, device="cuda"):
    sd = torch.load(os.path.join(ROOT, WEIGHTS[name]), map_location="cpu", weights_only=False)
    sd = sd.get("params_ema", sd.get("params", sd))
    m = RRDBNet()
    m.load_state_dict(sd, strict=True)
    return m.eval().to(device)


@torch.no_grad()
def sr_tiled(model, img_u8, tile=64, pad=24, device="cuda", half=True):
    """Super-resolve an HxWx3 uint8 image with overlapping tiles (keeps 4 GB GPUs happy)."""
    h, w, _ = img_u8.shape
    x = torch.from_numpy(img_u8).permute(2, 0, 1).float().div(255)[None].to(device)
    out = torch.zeros(1, 3, h * SCALE, w * SCALE, device=device)
    dt = torch.float16 if (half and device == "cuda") else torch.float32
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            ys, xs = max(y0 - pad, 0), max(x0 - pad, 0)
            ye, xe = min(y0 + tile + pad, h), min(x0 + tile + pad, w)
            with torch.autocast("cuda", dtype=dt, enabled=dt == torch.float16):
                o = model(x[:, :, ys:ye, xs:xe]).float()
            oy, ox = (y0 - ys) * SCALE, (x0 - xs) * SCALE
            th, tw = min(tile, h - y0) * SCALE, min(tile, w - x0) * SCALE
            out[:, :, y0 * SCALE:y0 * SCALE + th, x0 * SCALE:x0 * SCALE + tw] = o[:, :, oy:oy + th, ox:ox + tw]
    return (out.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def bicubic(img_u8):
    h, w = img_u8.shape[:2]
    return cv2.resize(img_u8, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def refl_to_u8(refl_rgb):
    """Surface reflectance (H,W,3 float) -> uint8 with a fixed linear stretch."""
    return (np.clip(np.nan_to_num(refl_rgb) / REFL_MAX, 0, 1) * 255).round().astype(np.uint8)


def read_lr_rgb(path):
    with rasterio.open(path) as d:
        a = d.read(out_dtype="float32")
    return refl_to_u8(a[RGB_IDX].transpose(1, 2, 0))


def hist_match(src, ref, mask):
    """Per-channel histogram matching of src to ref, statistics taken inside mask."""
    out = src.copy()
    for c in range(3):
        s = src[..., c][mask]
        r = ref[..., c][mask]
        s_vals, s_idx, s_cnt = np.unique(s, return_inverse=True, return_counts=True)
        r_vals, r_cnt = np.unique(r, return_counts=True)
        s_q = np.cumsum(s_cnt) / s.size
        r_q = np.cumsum(r_cnt) / r.size
        lut = np.interp(s_q, r_q, r_vals)
        full = np.interp(src[..., c].ravel(), s_vals, lut).reshape(src.shape[:2])
        out[..., c] = np.clip(full, 0, 255).round().astype(np.uint8)
    return out


def phase_shift(a_gray, b_gray, mask):
    """Sub-pixel (dx, dy) such that shifting b by it aligns b to a."""
    win = cv2.createHanningWindow(a_gray.shape[::-1], cv2.CV_32F)
    a = a_gray.astype(np.float32) * mask
    b = b_gray.astype(np.float32) * mask
    (dx, dy), resp = cv2.phaseCorrelate(b, a, win)
    return dx, dy, resp


def prepare_pair(lr_path, hr_path, border=16):
    """Build an aligned (LR uint8, HR-target uint8, valid mask) triplet at scale x4.

    HR (SPOT, 1.5 m) is area-resampled to exactly 4x the LR grid (2.5 m), histogram-matched
    to the bicubic-upsampled LR (removes sensor colour differences) and sub-pixel registered.
    """
    lr = read_lr_rgb(lr_path)
    h, w = lr.shape[:2]
    hr_rgba = cv2.imread(hr_path, cv2.IMREAD_UNCHANGED)
    hr = cv2.cvtColor(hr_rgba[..., :3], cv2.COLOR_BGR2RGB)
    alpha = hr_rgba[..., 3] if hr_rgba.shape[2] == 4 else np.full(hr.shape[:2], 255, np.uint8)
    H, W = h * SCALE, w * SCALE
    hr = cv2.resize(hr, (W, H), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(((alpha > 0) & (hr_rgba[..., :3].max(-1) > 0)).astype(np.uint8), (W, H),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    up = bicubic(lr)
    hr = hist_match(hr, up, valid)
    dx, dy, resp = phase_shift(cv2.cvtColor(up, cv2.COLOR_RGB2GRAY), cv2.cvtColor(hr, cv2.COLOR_RGB2GRAY),
                               valid.astype(np.float32))
    if abs(dx) < 12 and abs(dy) < 12:  # ignore implausible shifts (featureless scenes)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        hr = cv2.warpAffine(hr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        valid = cv2.warpAffine(valid.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST).astype(bool)
    else:
        dx = dy = 0.0
    valid[:border] = valid[-border:] = False
    valid[:, :border] = valid[:, -border:] = False
    return lr, hr, valid, (dx, dy, resp)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def psnr(a, b, mask):
    d = (a.astype(np.float64) - b.astype(np.float64))[mask]
    mse = np.mean(d ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-10))


def ssim(a, b, mask):
    from skimage.metrics import structural_similarity
    ys, xs = np.where(mask)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return structural_similarity(a[sl], b[sl], channel_axis=2, data_range=255)


_lpips = {}


def lpips_dist(a, b, mask, device="cuda"):
    import lpips
    if "net" not in _lpips:
        _lpips["net"] = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    ys, xs = np.where(mask)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    t = lambda im: torch.from_numpy(im[sl]).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(device)
    with torch.no_grad():
        return float(_lpips["net"](t(a), t(b)))


def lr_consistency(sr, lr):
    """PSNR between SR area-downsampled back to 10 m and the LR input: does SR respect the observation?"""
    h, w = lr.shape[:2]
    down = cv2.resize(sr, (w, h), interpolation=cv2.INTER_AREA)
    return psnr(down, lr, np.ones((h, w), bool))


def detail_energy(img, sigma=1.0, win=7):
    """Local high-frequency energy: |img - Gaussian(img)| averaged over a window (grayscale)."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), sigma))
    return cv2.blur(hp, (win, win))


def hallucination_maps(sr, hr, mask, tau=4.0):
    """Hallucination = structure in SR where the HR reference is flat; omission = the reverse.

    tau is in grey levels of local high-frequency energy. Returns maps and summary rates.
    """
    e_sr, e_hr = detail_energy(sr), detail_energy(hr)
    hall = (e_sr > tau) & (e_hr < tau / 2) & mask
    miss = (e_hr > tau) & (e_sr < tau / 2) & mask
    excess = np.clip(e_sr - e_hr, 0, None) * mask
    return {
        "hall_map": hall, "miss_map": miss, "excess": excess,
        "hall_rate": hall.sum() / mask.sum(), "miss_rate": miss.sum() / mask.sum(),
        "excess_detail": excess.sum() / mask.sum(),
    }


def correctness(sr, base, hr, mask, patch=8, min_ref=3.0):
    """Patch-wise improvement / omission / hallucination, simplified from OpenSR-test (Aybar et al. 2024).

    For every patch: d_ref = |bicubic - HR| (how much detail the LR is missing),
      d_im = |SR - HR| / d_ref   (<1: SR got closer to the truth than bicubic)
      d_om = |SR - bicubic| / d_ref (small: SR barely changed the bicubic image)
    improvement   : d_im < 1 and d_im <= d_om   -> SR added detail that is really there
    omission      : d_om < d_im and d_om < 1    -> SR stayed close to the blurry input
    hallucination : d_im >= 1 and d_om >= 1     -> SR moved away from BOTH LR and HR: invented content
    Patches where bicubic already matches HR (d_ref < min_ref grey levels) are skipped.
    """
    def pmean(x):
        h, w = x.shape[:2]
        hh, ww = h // patch * patch, w // patch * patch
        x = x[:hh, :ww]
        return x.reshape(hh // patch, patch, ww // patch, patch, *x.shape[2:]).mean(axis=(1, 3))

    f = lambda a, b: np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(-1)
    d_ref = pmean(f(base, hr))
    d_im = pmean(f(sr, hr)) / np.maximum(d_ref, 1e-6)
    d_om = pmean(f(sr, base)) / np.maximum(d_ref, 1e-6)
    valid = (pmean(mask.astype(np.float32)) > 0.99) & (d_ref >= min_ref)
    imp = valid & (d_im < 1) & (d_im <= d_om)
    hal = valid & (d_im >= 1) & (d_om >= 1)
    om = valid & ~imp & ~hal
    n = max(valid.sum(), 1)
    cls = np.full(d_ref.shape, -1, np.int8)
    cls[imp], cls[om], cls[hal] = 0, 1, 2
    return {"improvement": imp.sum() / n, "omission": om.sum() / n, "hallucination": hal.sum() / n, "cls": cls}


def edge_f1(sr, hr, mask, tol=2):
    """F1 between Canny edges of SR and HR with a tolerance of `tol` pixels (structure fidelity)."""
    e = lambda im: cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), (3, 3), 0), 50, 120) > 0
    es, eh = e(sr) & mask, e(hr) & mask
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    ds, dh = cv2.dilate(es.astype(np.uint8), k) > 0, cv2.dilate(eh.astype(np.uint8), k) > 0
    p = (es & dh).sum() / max(es.sum(), 1)
    r = (eh & ds).sum() / max(eh.sum(), 1)
    return 2 * p * r / max(p + r, 1e-9), p, r
