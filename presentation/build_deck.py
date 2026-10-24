"""Fill the presentation template (presentation/template_original.pptx) with the real results.

Keeps the template's design (Georgia/Arial, teal + gold palette, cards, labels) and replaces
placeholder text, empty image frames and "values pending" tables with numbers read from
results/metrics/*.csv at build time, so no number on a slide is typed by hand.

Usage:  python presentation/build_deck.py
Output: presentation/Super-Resolution_for_Port_Intelligence.pptx
"""
import copy
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
PRES = os.path.join(ROOT, "presentation")
MET = os.path.join(ROOT, "results", "metrics")
FIG = os.path.join(ROOT, "results", "figures")
TMP = os.path.join(PRES, "_assets")
os.makedirs(TMP, exist_ok=True)
SKILL_ADD_SLIDE = os.environ.get("ADD_SLIDE_SCRIPT")  # optional; we duplicate slides with python below instead

TEAL, GOLD, GREY, DARK = "0E6E75", "C98A2B", "6B7780", "1F2A33"

# ------------------------------------------------------------------ results
summ = pd.read_csv(os.path.join(MET, "per_tile_summary.csv")).set_index(["set", "method"])
ci = pd.read_csv(os.path.join(MET, "bootstrap_ci.csv")).set_index(["set", "method"])
ports = pd.read_csv(os.path.join(MET, "ports_summary.csv"), index_col=0)
per_tile = pd.read_csv(os.path.join(MET, "per_tile.csv"))
abl = {k: pd.read_json(os.path.join(ROOT, "results", "training", f"ablation_l1_{k}", "log.json"))
       for k in ["realesrgan", "satlas", "scratch"]}
abl_best = {k: v[v.it > 0].val_psnr.max() for k, v in abl.items()}
G, T = "gulf_test", "test"
n_ref_test = int(ci.loc[(T, "bicubic"), "det_ref_objects"])


def s(set_, m, col):
    return summ.loc[(set_, m), col]


# ------------------------------------------------------------------ helpers
def shape(slide, name):
    return next(sh for sh in slide.shapes if sh.name == name)


def set_paras(sh, paras):
    """Replace the text of a shape, keeping the template's paragraph and run formatting.

    paras: list of paragraphs; a paragraph is a str or a list of (text, bold) segments.
    Paragraph i is styled like template paragraph min(i, n-1) (e.g. label then body).
    """
    tf = sh.text_frame
    tpls = [copy.deepcopy(p._p) for p in tf.paragraphs]
    body = tf._txBody
    for p in list(tf.paragraphs):
        body.remove(p._p)
    for i, para in enumerate(paras):
        p = copy.deepcopy(tpls[min(i, len(tpls) - 1)])
        runs = p.findall(qn("a:r"))
        rprs = [r.find(qn("a:rPr")) for r in runs]
        plain = isinstance(para, str)
        if plain:  # keep template formatting exactly (bold card titles stay bold)
            base = rprs[0] if rprs else None
        else:  # segments: start from a non-bold run and set bold per segment
            base = next((r for r in rprs if r is not None and r.get("b") != "1"), rprs[0] if rprs else None)
        for el in list(p):
            if el.tag in (qn("a:r"), qn("a:br"), qn("a:fld")):
                p.remove(el)
        end = p.find(qn("a:endParaRPr"))
        segs = [(para, False)] if isinstance(para, str) else para
        for text, bold in segs:
            r = etree.SubElement(p, qn("a:r"))
            if base is not None:
                rp = copy.deepcopy(base)
                if not plain:
                    if bold:
                        rp.set("b", "1")
                    elif "b" in rp.attrib:
                        del rp.attrib["b"]
                r.append(rp)
            t = etree.SubElement(r, qn("a:t"))
            t.text = text
            if end is not None:
                end.addprevious(r)
        body.append(p)


def set_cell(cell, text, bold=None):
    p = cell.text_frame.paragraphs[0]
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    if bold is not None:
        runs[0].font.bold = bold
    for extra in cell.text_frame.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)


def resize_rows(table, n_rows, src_rows):
    """Grow/shrink the table to n_rows; new rows are copies of src_rows[i] (template row indices)."""
    tbl = table._tbl
    trs = tbl.tr_lst
    keep = [copy.deepcopy(trs[i]) for i in src_rows]
    for tr in trs[1:]:
        tbl.remove(tr)
    for tr in keep[:n_rows - 1]:
        tbl.append(tr)


def crop_to(img, w_in, h_in, center=None):
    """Crop an RGB array to the aspect ratio of a w_in x h_in frame (around center if given)."""
    H, W = img.shape[:2]
    target = w_in / h_in
    if W / H > target:
        nw, nh = int(H * target), H
    else:
        nw, nh = W, int(W / target)
    cy, cx = center if center is not None else (H // 2, W // 2)
    y0 = int(np.clip(cy - nh // 2, 0, H - nh))
    x0 = int(np.clip(cx - nw // 2, 0, W - nw))
    return img[y0:y0 + nh, x0:x0 + nw]


def put_image(slide, frame_name, img, fname, remove_frame=True, box=None, fit=False):
    """Place an RGB array exactly where a template image frame was (or in box=(x,y,w,h) inches)."""
    if box is None:
        fr = shape(slide, frame_name)
        x, y, w, h = fr.left, fr.top, fr.width, fr.height
        if remove_frame:
            fr._element.getparent().remove(fr._element)
    else:
        x, y, w, h = [Inches(v) for v in box]
    if fit:  # keep the whole image, centred in the box
        r = img.shape[1] / img.shape[0]
        if w / h > r:
            nw = int(h * r)
            x, w = x + (w - nw) // 2, nw
        else:
            nh = int(w / r)
            y, h = y + (h - nh) // 2, nh
    else:
        img = crop_to(img, w, h)
    path = os.path.join(TMP, fname)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return slide.shapes.add_picture(path, x, y, w, h)


def structure_center(img, win=200):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    e = cv2.blur(np.abs(cv2.Laplacian(g, cv2.CV_32F)), (win, win))
    return np.unravel_index(np.argmax(e), e.shape)


def remove(slide, *names):
    for nm in names:
        el = shape(slide, nm)._element
        el.getparent().remove(el)


def rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def gamma(img, g=0.6):
    return (255 * (img / 255.0) ** g).astype(np.uint8)


def duplicate_slide(prs, index):
    """Duplicate slide `index` (0-based) at the end, copying shapes and image relationships."""
    src = prs.slides[index]
    dst = prs.slides.add_slide(src.slide_layout)
    for ph in list(dst.placeholders):
        ph._element.getparent().remove(ph._element)
    for el in src.shapes._spTree.iterchildren():
        if el.tag.endswith("}nvGrpSpPr") or el.tag.endswith("}grpSpPr"):
            continue
        dst.shapes._spTree.append(copy.deepcopy(el))
    for rel in src.part.rels.values():
        if "notesSlide" in rel.reltype:
            continue
        if rel.reltype.endswith("/image"):
            dst.part.rels.get_or_add(rel.reltype, rel._target)
    return dst


def move_slide(prs, old, new):
    ids = prs.slides._sldIdLst
    el = list(ids)[old]
    ids.remove(el)
    ids.insert(new, el)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def style_chart(chart, size=11):
    chart.font.size = Pt(size)
    chart.font.name = "Arial"
    chart.font.color.rgb = RGBColor.from_string(DARK)


# ------------------------------------------------------------------ build
prs = Presentation(os.path.join(PRES, "template_original.pptx"))
# structure first: 8 template slides -> 14. New slides are copies of template slides, so the design stays identical.
data_slide = duplicate_slide(prs, 5)      # 4 image frames -> data sources
abl_slide = duplicate_slide(prs, 1)       # two columns    -> initialisation ablation
gallery_slide = duplicate_slide(prs, 5)   # title only     -> all methods on Gulf tiles
portview_slide = duplicate_slide(prs, 5)  # title only     -> port views with detections
port_slide = duplicate_slide(prs, 1)      # two columns    -> vessels found after SR
end_slide = duplicate_slide(prs, 0)       # dark           -> conclusions
s1, s2, s3, s4, s5, s6, s7, s8 = [prs.slides[i] for i in range(8)]
c_l1 = ci.loc[(G, "ft_l1")]
d_scr = abl_best["realesrgan"] - abl_best["scratch"]

# ---------------- 1. title
set_paras(shape(s1, "Subtitle 2"), ["An evaluation of whether 10 m → 2.5 m super-resolution of Sentinel-2 imagery supports the "
                                     "analysis of UAE ports, and where it introduces detail that does not exist"])
notes(s1, "This project evaluates whether super-resolution of free 10 m Sentinel-2 imagery supports the analysis of UAE ports, "
          "and measures where it introduces detail that is not present in reality.\n\nIt is the second of three projects under "
          "'A Gulf Maritime Picture': (1) dark-vessel detection from SAR and AIS, (2) this study, and (3) coastal bathymetry.\n\n"
          "Summary of findings: the training domain matters more than the architecture; fidelity-oriented super-resolution is "
          "reliable but yields only a small gain; GAN-based super-resolution produces realistic imagery and makes some vessels "
          "detectable, but its synthesised texture must not be treated as evidence.")

# ---------------- 2. problem (template content kept, image added)
kh = rgb(os.path.join(ROOT, "data", "s2_ports", "khalifa_port", "2025-11-01_rgb.png"))
kcy, kcx = structure_center(kh, 120)
fr = shape(s2, "Image Frame Port")
kh_c = crop_to(kh, fr.width, fr.height, center=(kcy, kcx))
kh_c = cv2.resize(kh_c, (kh_c.shape[1] * 4, kh_c.shape[0] * 4), interpolation=cv2.INTER_LANCZOS4)
put_image(s2, "Image Frame Port", gamma(kh_c), "khalifa_s2.png")
notes(s2, "Sentinel-2 provides a free image of every UAE port every few days, but at 10 m a container vessel spans roughly 30 "
          "pixels, and cranes or small craft are below the pixel size.\n\nSuper-resolution is a listed capability of the Stella "
          "platform, and Orbion addresses maritime and port intelligence. The risk is that GAN-based super-resolution draws "
          "structures that do not exist; in an intelligence context an invented vessel is more harmful than a blurred one, because "
          "it creates unwarranted confidence.\n\nImage: Khalifa Port, Sentinel-2 L2A, 1 November 2025, 10 m (brightened for display).")

# ---------------- 3. data (new, images)
set_paras(shape(data_slide, "Title 1"), ["Paired imagery is aligned before any comparison is made"])
probe = os.path.join(ROOT, "data", "raw", "probe")
from srlib import prepare_pair  # noqa: E402
lr_p, hr_p, _, _ = prepare_pair(os.path.join(probe, "Landcover-773487-1-L2A_data.tiff"), os.path.join(probe, "Landcover-773487_rgb.png"))
spot_raw = cv2.cvtColor(cv2.imread(os.path.join(probe, "Landcover-773487_rgb.png"), cv2.IMREAD_UNCHANGED)[..., :3], cv2.COLOR_BGR2RGB)
lr_p_up = cv2.resize(lr_p, (hr_p.shape[1], hr_p.shape[0]), interpolation=cv2.INTER_NEAREST)
spot_raw_r = cv2.resize(spot_raw, (hr_p.shape[1], hr_p.shape[0]), interpolation=cv2.INTER_AREA)
pc = lambda im: im[180:480, 200:440]  # noqa: E731  same crop, portrait
kh_full = cv2.resize(kh, (kh.shape[1] * 2, kh.shape[0] * 2), interpolation=cv2.INTER_LANCZOS4)
for i, (c1, c2, im) in enumerate([
        ("1  SENTINEL-2 L2A", "Model input, 10 m", pc(lr_p_up)),
        ("2  SPOT 6/7 AS DELIVERED", "1.5 m, different sensor colours", pc(spot_raw_r)),
        ("3  SPOT AFTER ALIGNMENT", "2.5 m, matched and registered", pc(hr_p)),
        ("4  UAE PORT AREA OF INTEREST", "Khalifa Port, Sentinel-2 only", gamma(crop_to(kh_full, 2.83, 3.56, center=(kcy * 2, kcx * 2))))]):
    set_paras(shape(data_slide, f"Caption {i}"), [c1, c2])
    put_image(data_slide, f"Image Frame {i}", im, f"data_{i}.png")
set_paras(shape(data_slide, "Note"), ["Panels 1–3: WorldStrat tile Landcover-773487 (Sharjah), held out from training. The SPOT reference is "
                                      "resampled to 2.5 m, histogram-matched to Sentinel-2 and sub-pixel registered by phase correlation, "
                                      "so that metrics measure spatial detail rather than sensor calibration."])
notes(data_slide, "Two data sources. WorldStrat (NeurIPS 2022) pairs real Sentinel-2 revisits with real 1.5 m SPOT imagery "
                  "worldwide, including the Gulf. The full dataset is 104 GB; I read the zip index with HTTP range requests and "
                  "downloaded approximately 2.5 GB: 600 training, 60 validation and 120 test tiles, plus all 13 Gulf tiles held out.\n\n"
                  "Panel 2 shows why alignment is required: SPOT colours differ strongly from Sentinel-2. Without colour matching, "
                  "PSNR would primarily measure the sensor difference. I also register the images to sub-pixel accuracy, since a "
                  "one-pixel offset at 2.5 m costs several dB and penalises sharp methods.\n\nFor the UAE ports themselves no free "
                  "2.5 m reference exists, so they are evaluated with reference-free checks later in the deck.")

# ---------------- 4. pipeline
set_paras(shape(s3, "Title 1"), ["Five upscaling methods pass through one evaluation pipeline"])
for name, title, body in [
        ("D1", "WorldStrat v1.1", "Sentinel-2 and SPOT 1.5 m pairs; 600 train, 60 val, 120 test, 13 Gulf tiles held out"),
        ("D2", "Sentinel-2 L2A", "Khalifa, Jebel Ali and Zayed Port; four cloud-free dates each"),
        ("M0", "Bicubic", "Interpolation baseline"),
        ("M1", "Pre-trained", "Real-ESRGAN (photographs) and Satlas (Sentinel-2 to US aerial)"),
        ("M2", "Fine-tuned on WorldStrat", "L1 loss (fidelity) and L1 + VGG + GAN (realism)"),
        ("V0", "Image quality", "PSNR, SSIM, LPIPS against SPOT at 2.5 m"),
        ("V1", "Usefulness", "Frozen YOLO11-OBB (DOTA) on SR versus on high-res"),
        ("V2", "Hallucination", "Patch correctness, input consistency, stability across dates"),
        ("O0", "Comparison", "Five methods with bootstrap 95 % confidence intervals"),
        ("O1", "Before / after demo", "Streamlit slider over UAE ports"),
        ("O2", "Failure gallery", "Where SR introduces detail")]:
    set_paras(shape(s3, name), [title, body])
notes(s3, "Every method passes through the same pipeline. Data: WorldStrat for training and reference-based evaluation; "
          "Sentinel-2 L2A over the three ports from Element 84 Earth Search (AWS open data, no account required, reflectance "
          "offset of -0.1 handled).\n\nEvaluation on three axes: image quality, usefulness for a frozen detector, and hallucination.")

# ---------------- 5. build vs reuse
set_paras(shape(s4, "Title 1"), ["Reuse pre-trained weights, then fine-tune on real satellite pairs"])
tb = shape(s4, "Build vs Reuse Table")
resize_rows(tb.table, 6, [1, 2, 2, 3, 3])
rows = [("Bicubic", "Standard library", "Sets the floor that every learned model must exceed"),
        ("Real-ESRGAN, pre-trained", "Reuse", "General-purpose SR trained on degraded photographs"),
        ("Satlas, pre-trained", "Reuse", "Same network trained on Sentinel-2 to US aerial imagery"),
        ("Fine-tuned L1", "Fine-tune", "Real-ESRGAN weights, 4,000 iterations, L1 loss: fidelity"),
        ("Fine-tuned GAN", "Fine-tune", "From the L1 model, adds perceptual and GAN loss: realism")]
for i, r in enumerate(rows, start=1):
    for j, v in enumerate(r):
        set_cell(tb.table.cell(i, j), v)
for i, row in enumerate(tb.table.rows):
    row.height = Inches(0.45 if i == 0 else 0.42)
for nm in ["Scratch", "Scratch Accent", "Fallback", "Fallback Accent"]:
    shape(s4, nm).top = Inches(4.95)
set_paras(shape(s4, "Scratch"), ["WHY NOT TRAIN FROM SCRATCH",
                                 f"Measured rather than assumed: with identical data, loss and 1,500 iterations, a random initialisation "
                                 f"reaches {abl_best['scratch']:.1f} dB validation PSNR, against {abl_best['realesrgan']:.1f} dB from "
                                 f"Real-ESRGAN weights."])
set_paras(shape(s4, "Fallback"), ["WHY NOT DIFFUSION OR MULTI-IMAGE SR",
                                  "Diffusion models exceed the 4 GB GPU budget and hallucinate more in published benchmarks. Multi-image "
                                  "SR fails on vessels that move between passes; it is the next step for static infrastructure."])
notes(s4, "Both pre-trained models share exactly the same RRDBNet architecture (the ESRGAN generator: 23 blocks, 16.7 million "
          "parameters), so comparing them isolates the effect of training data. I re-implemented RRDBNet in about 100 lines only "
          "because the BasicSR package is incompatible with current torchvision, and verified that it loads both official "
          "checkpoints with strict key matching.\n\nTraining: 4,000 L1 iterations from Real-ESRGAN weights (selected by the "
          "ablation on the next slide), then 3,000 GAN iterations. Adam optimiser, cosine learning-rate decay, exponential moving "
          "average of weights, full precision because half precision produced NaN losses on the GTX 1650.")

# ---------------- 6. initialisation ablation
set_paras(shape(abl_slide, "Title 1"), [f"Pre-trained weights outperform training from scratch by {d_scr:.0f} dB on the same budget"])
set_paras(shape(abl_slide, "Question Label"), ["VALIDATION PSNR DURING TRAINING, 30 VALIDATION TILES (dB)"])
remove(abl_slide, "Question Body", "Image Frame Port")
cd = CategoryChartData()
cd.categories = [str(int(i)) for i in abl["realesrgan"].it]
for k, lab in [("realesrgan", "Initialised from Real-ESRGAN"), ("satlas", "Initialised from Satlas"), ("scratch", "Random initialisation")]:
    cd.add_series(lab, [round(v, 2) for v in abl[k].val_psnr])
ch = abl_slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.4), Inches(2.3), Inches(6.7), Inches(4.4), cd).chart
style_chart(ch)
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
for ser, col in zip(ch.series, [TEAL, GOLD, GREY]):
    ser.format.line.color.rgb = RGBColor.from_string(col)
    ser.format.line.width = Pt(2.25)
    ser.smooth = False
    ser.marker.format.fill.solid()
    ser.marker.format.fill.fore_color.rgb = RGBColor.from_string(col)
    ser.marker.format.line.color.rgb = RGBColor.from_string(col)
va = ch.value_axis
va.minimum_scale, va.maximum_scale = 12, 30
va.major_gridlines.format.line.color.rgb = RGBColor.from_string("E3E6E6")
va.format.line.fill.background()
ch.category_axis.format.line.color.rgb = RGBColor.from_string("9FB4B8")
ch.category_axis.has_title = True
ch.category_axis.axis_title.text_frame.text = "Training iteration"
ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.bold = False
set_paras(shape(abl_slide, "Relevance Label"), ["WHAT THE ABLATION SHOWS"])
set_paras(shape(abl_slide, "Relevance Body"), [
    [("Identical data, loss and 1,500 iterations; ", False), ("only the initial weights differ", True)],
    [("Real-ESRGAN: ", True), (f"{abl_best['realesrgan']:.1f} dB; starts high and continues to improve", False)],
    [("Satlas: ", True), (f"{abl_best['satlas']:.1f} dB; satellite weights, but trained on US imagery only", False)],
    [("Random: ", True), (f"{abl_best['scratch']:.1f} dB and still rising; would require far more data and compute", False)],
    [("Decision: ", True), ("fine-tune from Real-ESRGAN weights", False)]])
notes(abl_slide, "This addresses the question from the first interview on pre-trained models. Identical data (1,607 pairs), "
                 "loss and number of iterations; only the initial weights differ.\n\n"
                 f"Real-ESRGAN initialisation ends at {abl_best['realesrgan']:.2f} dB, Satlas at {abl_best['satlas']:.2f} dB, random at "
                 f"{abl_best['scratch']:.2f} dB. Notably, the satellite-specific Satlas weights perform worse than generic photographic "
                 "weights: Satlas was trained on US imagery with NAIP colours and must first unlearn that prior on Gulf scenes. Generic "
                 "weights provide useful low-level features (edges, textures) without an incorrect geographic prior.\n\n"
                 "Source: results/training/ablation_l1_*/log.json.")

# ---------------- 7. results
set_paras(shape(s5, "Title 1"), ["Only the L1 model improves fidelity; the GAN model improves perceptual quality"])
tb = shape(s5, "Results Table")
resize_rows(tb.table, 6, [1, 2, 2, 3, 3])
for j, v in enumerate(["Method", "PSNR (dB) ↑  Gulf / global", "SSIM ↑", "LPIPS ↓", "Hallucinated patches ↓"]):
    set_cell(tb.table.cell(0, j), v)
names = [("bicubic", "Bicubic"), ("realesrgan", "Real-ESRGAN, pre-trained"), ("satlas", "Satlas, pre-trained"),
         ("ft_l1", "Fine-tuned L1"), ("ft_gan", "Fine-tuned GAN")]
for i, (m, lab) in enumerate(names, start=1):
    vals = [lab, f"{s(G, m, 'psnr'):.2f} / {s(T, m, 'psnr'):.2f}", f"{s(G, m, 'ssim'):.3f}", f"{s(G, m, 'lpips'):.3f}",
            f"{100 * s(G, m, 'hallucination'):.1f} %"]
    for j, v in enumerate(vals):
        set_cell(tb.table.cell(i, j), v)
for i, row in enumerate(tb.table.rows):
    row.height = Inches(0.5 if i == 0 else 0.42)
set_paras(shape(s5, "Q"), ["Image quality",
                           f"Fine-tuned L1 exceeds bicubic by {c_l1.d_psnr:.2f} dB on Gulf tiles (95 % CI {c_l1.d_psnr_lo:.2f} to "
                           f"{c_l1.d_psnr_hi:.2f}): significant but small. All GAN models trade PSNR for LPIPS."])
set_paras(shape(s5, "U"), ["Usefulness",
                           f"Detector agreement with SPOT is inconclusive: {n_ref_test} reference objects in 120 test tiles, and all "
                           f"F1 intervals overlap (bicubic {ci.loc[(T, 'bicubic'), 'det_f1']:.2f}, GAN {ci.loc[(T, 'ft_gan'), 'det_f1']:.2f})."])
set_paras(shape(s5, "H"), ["Hallucination",
                           f"Satlas introduces structure in {100 * s(G, 'satlas', 'hallucination'):.0f} % of Gulf patches. Fine-tuning "
                           f"reduces Real-ESRGAN's {100 * s(T, 'realesrgan', 'hallucination'):.0f} % to {100 * s(T, 'ft_gan', 'hallucination'):.0f} % "
                           f"(GAN) and {100 * s(T, 'ft_l1', 'hallucination'):.0f} % (L1) on the global test set."])
set_paras(shape(s5, "Footnote"), ["13 held-out Gulf tiles (never used in training) / 120 global WorldStrat test tiles. SSIM, LPIPS and "
                                  "hallucination on Gulf tiles. SR colours harmonised to the input before scoring; confidence intervals "
                                  "from 10,000 paired bootstrap resamples. Source: results/metrics/per_tile_summary.csv, bootstrap_ci.csv."])
notes(s5, "These results describe a trade-off rather than a single ranking.\n\n"
          f"Fidelity: only the L1 fine-tuned model exceeds bicubic, by {c_l1.d_psnr:.2f} dB on Gulf tiles. The bootstrap interval "
          "excludes zero, so the gain is genuine, but it is small. This is consistent with the literature: super-resolution "
          "between real Sentinel-2 and SPOT imagery is considerably harder than on synthetic benchmarks.\n\n"
          "Perceptual quality: the GAN model has the best LPIPS by a wide margin but lower PSNR: the perception-distortion "
          "trade-off described by Blau and Michaeli (2018).\n\nHallucination: measured per patch, simplified from ESA's OpenSR-test; "
          "a patch is counted as hallucinated when the output moves away from both the input and the reference.\n\n"
          "Usefulness: the downstream detection test is underpowered, because WorldStrat tiles contain few detectable objects.")

# ---------------- 8. gallery: all methods on held-out Gulf tiles (new, image)
set_paras(shape(gallery_slide, "Title 1"), ["All five methods on held-out Gulf tiles, against the SPOT reference"])
for i in range(4):
    remove(gallery_slide, f"Caption {i}", f"Image Frame {i}")
gal = rgb(os.path.join(FIG, "examples_gulf_slide.png"))  # slide version: 3 tiles, larger labels
put_image(gallery_slide, None, gal, "gallery_gulf.png", box=(0.5, 1.5, 12.33, 4.75), fit=True)
set_paras(shape(gallery_slide, "Note"), ["Three of the 13 held-out Gulf tiles. Under each panel: PSNR against the reference and the share "
                                         "of hallucinated patches. Satlas renders vegetation typical of its US training data; the GAN "
                                         "model reconstructs the street layout most convincingly."])
shape(gallery_slide, "Note").top = Inches(6.35)
notes(gallery_slide, "This gallery shows every method on held-out Gulf tiles. Three observations.\n\nFirst, Satlas consistently "
                     "replaces Gulf scenes with vegetation and terrain typical of its US training data; this is the domain-shift failure.\n\n"
                     "Second, the L1 model is faithful but remains soft; visually it is close to bicubic.\n\nThird, the GAN model "
                     "reconstructs street grids and building outlines that resemble the reference, yet its PSNR is lower, because the "
                     "individual structures are placed slightly incorrectly. Source: results/figures/examples_gulf.png.")

# ---------------- 9. same tile under each method
tile = "Landcover-773487"  # held-out Gulf tile, Sharjah (25.35 N, 55.47 E)
d = os.path.join(ROOT, "results", "sr_outputs", G, tile)
hr = rgb(os.path.join(d, "hr.png"))
lr_up = cv2.resize(rgb(os.path.join(d, "lr.png")), (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_NEAREST)
ims = {"lr": lr_up, "ft_l1": rgb(os.path.join(d, "ft_l1.png")), "ft_gan": rgb(os.path.join(d, "ft_gan.png")), "hr": hr}
crop = lambda im: im[180:480, 210:450]  # noqa: E731  portrait crop, same in every panel
row = per_tile[(per_tile.tile == tile)].set_index("method")
set_paras(shape(s6, "Title 1"), ["A closer view: one Sharjah tile, fidelity model versus GAN model"])
caps = [("1  INPUT", "Sentinel-2, 10 m"),
        ("2  FINE-TUNED L1", f"{row.loc['ft_l1', 'psnr']:.1f} dB, {100 * row.loc['ft_l1', 'hallucination']:.0f} % hallucinated"),
        ("3  FINE-TUNED GAN", f"{row.loc['ft_gan', 'psnr']:.1f} dB, {100 * row.loc['ft_gan', 'hallucination']:.0f} % hallucinated"),
        ("4  REFERENCE", "SPOT, resampled to 2.5 m")]
for i, (k, (c1, c2)) in enumerate(zip(["lr", "ft_l1", "ft_gan", "hr"], caps)):
    set_paras(shape(s6, f"Caption {i}"), [c1, c2])
    put_image(s6, f"Image Frame {i}", crop(ims[k]), f"sharjah_{k}.png")
set_paras(shape(s6, "Note"), [f"Bicubic on this tile: {row.loc['bicubic', 'psnr']:.1f} dB. Sentinel-2 L2A via WorldStrat; reference: "
                              "WorldStrat SPOT 6/7 (CC BY-NC 4.0). Tile Landcover-773487, never used in training. Same crop in every panel."])
notes(s6, "A dense residential district in Sharjah that the models never saw during training.\n\nThe L1 model is faithful but soft. "
          "The GAN model produces a convincing street grid and building outlines that resemble the reference, yet its PSNR is "
          "lower, because individual roofs and walls are placed slightly incorrectly. This is the perception-distortion trade-off "
          "in a single example, and the reason an analyst should not count buildings or vehicles on GAN output.")

# ---------------- 10. hallucination
set_paras(shape(s7, "Title 1"), ["Hallucinations are identified by comparing SR with both the input and the reference"])
for i, (a, b) in enumerate([("Align", "Colour-match and sub-pixel register SPOT to Sentinel-2"),
                            ("Compare", "Per 8×8 patch: distance of SR to the reference and to bicubic"),
                            ("Flag", "Hallucinated: SR moved away from both input and reference"),
                            ("Review", "Zoomed crops and difference maps of the worst tiles")]):
    set_paras(shape(s7, f"Step {i}"), [a, b])
set_paras(shape(s7, "Lesson Label"), ["WHAT THE FAILURE CASES SHOW"])
set_paras(shape(s7, "Lessons"), [
    [("Domain shift: ", True), (f"Satlas synthesises vegetation from its US training data ({100 * s(G, 'satlas', 'hallucination'):.0f} % of Gulf patches)", False)],
    [("GAN texture ", True), ("is realistic but not accurate: roofs and walls are slightly misplaced", False)],
    [("Some flags reflect real change ", True), ("between acquisition dates, such as a replanted field", False)]])
hm = rgb(os.path.join(FIG, "hallucination_map_gulf_ft_gan.png"))
hm = hm[:, int(hm.shape[1] * 0.6):]  # difference map + correctness overlay
put_image(s7, "Image Frame Failure", hm, "hallucination_panels.png")
notes(s7, "Method: after colour matching and registration, for each 8x8 patch I compute d_im = |SR - HR| / |bicubic - HR| and "
          "d_om = |SR - bicubic| / |bicubic - HR|. If the output is closer to the reference than bicubic, the patch is an "
          "improvement (blue). If it has moved away from both the input and the reference, it is a hallucination (red). This is a "
          "simplified version of the correctness metric in ESA's OpenSR-test benchmark.\n\nImage: left, absolute difference to the "
          "reference; right, the patch classification. The bright rectangle in the difference map is a field that changed between "
          "the SPOT and Sentinel-2 acquisitions: genuine change rather than hallucination, which is the main limitation of any "
          "reference-based check.")

# ---------------- 11. port views (new, images)
set_paras(shape(portview_slide, "Title 1"), ["Super-resolution over the UAE ports, with detector output overlaid"])
for i in range(4):
    remove(portview_slide, f"Caption {i}", f"Image Frame {i}")
for port, y in [("khalifa_port", 1.45), ("jebel_ali_port", 3.9)]:
    im = rgb(os.path.join(FIG, f"port_{port}_slide.png"))
    put_image(portview_slide, None, im, f"portview_{port}.png", box=(0.5, y, 12.33, 2.35), fit=True)
set_paras(shape(portview_slide, "Note"), ["Sentinel-2 L2A (Element 84 Earth Search), ×4. Boxes: frozen YOLO11-OBB trained on DOTA "
                                          "(cyan: ship; yellow: other classes such as storage tanks). Display brightened identically "
                                          "in every panel. Counts refer to the full 6 × 6 km area of interest."])
shape(portview_slide, "Note").top = Inches(6.4)
notes(portview_slide, "Two of the three ports, each shown with all five methods and the detector output. At Jebel Ali, the "
                      "storage-tank farm is detected consistently by every method except Satlas, which converts the port into "
                      "US-style terrain and suppresses most detections. The fine-tuned models retain the port structure and add "
                      "sharper edges. No free 2.5 m reference exists for these dates, so the next slide uses reference-free checks.")

# ---------------- 12. ports: vessels found after SR
P = ports
set_paras(shape(port_slide, "Title 1"), ["GAN super-resolution makes berthed vessels detectable, but its deck detail is not evidence"])
set_paras(shape(port_slide, "Question Label"), ["VESSELS DETECTED ONLY AFTER GAN SR ARE VISIBLE IN THE 10 M INPUT"])
remove(port_slide, "Question Body", "Image Frame Port")
put_image(port_slide, None, rgb(os.path.join(FIG, "port_gan_only_ships_2.png")), "port_ships.png", box=(0.5, 2.36, 6.53, 4.4), fit=True)
set_paras(shape(port_slide, "Relevance Label"), ["REFERENCE-FREE CHECKS: 3 PORTS × 4 DATES"])
set_paras(shape(port_slide, "Relevance Body"), [
    [("Vessels per image: ", True), (f"bicubic {P.loc['bicubic', 'n_ship']:.1f}, GAN {P.loc['ft_gan', 'n_ship']:.1f}, "
                                     f"L1 {P.loc['ft_l1', 'n_ship']:.1f}, Real-ESRGAN {P.loc['realesrgan', 'n_ship']:.1f}", False)],
    [("The additional vessels are real: ", True), ("they are visible in the Sentinel-2 input", False)],
    [("Satlas suppresses port structure: ", True), (f"{P.loc['satlas', 'n_det']:.1f} detections per image, against {P.loc['bicubic', 'n_det']:.0f} for bicubic", False)],
    [("GAN detail varies between dates: ", True), (f"+{P.loc['ft_gan', 'excess_instability']:.1f} grey levels on pixels the input shows as unchanged", False)],
    [("Implication: ", True), ("detections may be used; synthesised texture may not", False)]])
notes(port_slide, "As no 2.5 m reference exists for these dates, the ports are evaluated with reference-free checks.\n\n"
                  "1) Detections: the frozen DOTA detector finds more vessels after GAN super-resolution. I inspected every vessel "
                  "that appears only after GAN SR: all are genuine berthed vessels, visible in the 10 m pixels. Super-resolution "
                  "therefore assists the detector. However, the containers and superstructure drawn on deck cannot be verified from "
                  "10 m data.\n\n2) Temporal stability: between two dates, on pixels where the input is unchanged, the output should "
                  "also be unchanged. The GAN output varies most, which indicates that part of its fine detail is generated rather than "
                  "observed.\n\n3) Satlas suppresses the port structure entirely.\n\nSource: results/metrics/ports_summary.csv.")

# ---------------- 13. limitations
set_paras(shape(s8, "Lim Body"), [
    [("SPOT and Sentinel-2 are ", False), ("not acquired simultaneously", True), (", so some flags are real change", False)],
    [("The detection test is ", False), (f"underpowered ({n_ref_test} objects)", True), ("; labels are pseudo ground truth", False)],
    [("A linear 8-bit stretch ", False), ("darkens water", True), ("; WorldStrat SPOT data is CC BY-NC", False)]])
for nm, y in [("Next Label", 4.15), ("Next Body", 4.51)]:
    shape(s8, nm).top = Inches(y)
set_paras(shape(s8, "Next Body"), [
    [("Multi-image SR", True), (" from repeated passes for static port infrastructure", False)],
    [("SEN2SR", True), (" (radiometrically consistent, ESA OpenSR) as a further baseline", False)],
    [("A hallucination map", True), (" delivered with every SR output", False)]])
set_paras(shape(s8, "LText 1"), ["Super-Resolution for Port Intelligence", "Sharper views, delivered with a hallucination check"])
notes(s8, "Limitations: the acquisition gap between SPOT and Sentinel-2, the small number of detectable objects in the test "
          "tiles, and the non-commercial licence of the reference imagery, which is suitable for research but not for a product.\n\n"
          "Position in the wider picture: project 1 identifies dark vessels from SAR and AIS; super-resolution could support an "
          "analyst inspecting optical image chips of those candidates, provided each output carries a hallucination flag. "
          "Project 3, bathymetry, should not use super-resolved imagery as input, because depth retrieval depends on the "
          "original reflectance values.")

# ---------------- 14. conclusions (dark closing slide)
set_paras(shape(end_slide, "Title 1"), ["Conclusions"])
sub = shape(end_slide, "Subtitle 2")
sub.top, sub.height = Inches(4.3), Inches(1.7)
set_paras(sub, ["1   The training domain matters more than the architecture: US-only pretraining fails in the Gulf",
                f"2   Fidelity-oriented SR is reliable but gains only {c_l1.d_psnr:.2f} dB; GAN SR aids vessel detection",
                "3   SR should be used as a visual aid, with every output accompanied by a hallucination check"])
for p in sub.text_frame.paragraphs:
    for r in p.runs:
        r.font.size = Pt(18)
set_paras(shape(end_slide, "Presenter"), ["Demonstration: streamlit run demo/app.py  |  Code, data scripts and results in the project repository  |  Questions"])
shape(end_slide, "Presenter").top = Inches(6.35)
notes(end_slide, "Conclude with the three findings, then switch to the live demonstration (before/after slider over Khalifa Port, "
                 "Jebel Ali and Zayed Port) and take questions.")

# final order: title, problem, data, pipeline, build/reuse, ablation, results, gallery, close-up, hallucination,
#              port views, vessels, limitations, conclusions
order_src = [0, 1, 8, 2, 3, 9, 4, 10, 5, 6, 11, 12, 7, 13]
ids = list(prs.slides._sldIdLst)
for el in ids:
    prs.slides._sldIdLst.remove(el)
for i in order_src:
    prs.slides._sldIdLst.append(ids[i])

out = os.path.join(PRES, "Super-Resolution_for_Port_Intelligence.pptx")
prs.save(out)
print("saved", out)
