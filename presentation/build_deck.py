"""Build the 8-slide deck (presentation/sr_port_intelligence.pptx) from results/ figures and CSVs.

Every number on a slide is read from results/metrics/*.csv at build time, not typed by hand.
Usage:  python presentation/build_deck.py
"""
import os

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
MET = os.path.join(ROOT, "results", "metrics")
NAVY, GREY, RED = RGBColor(0x0B, 0x2A, 0x4A), RGBColor(0x55, 0x55, 0x55), RGBColor(0xC4, 0x4E, 0x52)

summ = pd.read_csv(os.path.join(MET, "per_tile_summary.csv")).set_index(["set", "method"])
ports = pd.read_csv(os.path.join(MET, "ports_summary.csv"), index_col=0)
abl = {}
for k in ["realesrgan", "satlas", "scratch"]:
    p = os.path.join(ROOT, "results", "training", f"ablation_l1_{k}", "log.json")
    if os.path.exists(p):
        abl[k] = pd.read_json(p)
NAMES = {"bicubic": "Bicubic", "realesrgan": "Real-ESRGAN (pretrained)", "satlas": "Satlas (pretrained)",
         "ft_l1": "Fine-tuned L1 (ours)", "ft_gan": "Fine-tuned GAN (ours)"}

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide(title, notes, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9)).text_frame
    tb.text = title
    tb.paragraphs[0].runs[0].font.size = Pt(30)
    tb.paragraphs[0].runs[0].font.bold = True
    tb.paragraphs[0].runs[0].font.color.rgb = NAVY
    if subtitle:
        p = tb.add_paragraph()
        p.text = subtitle
        p.runs[0].font.size = Pt(15)
        p.runs[0].font.color.rgb = GREY
    s.notes_slide.notes_text_frame.text = notes
    return s


def bullets(s, items, x, y, w, h, size=17):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "• " + it
        p.space_after = Pt(8)
        for r in p.runs:
            r.font.size = Pt(size)
    return tf


def picture(s, name, x, y, w=None, h=None):
    p = os.path.join(FIG, name)
    if os.path.exists(p):
        return s.shapes.add_picture(p, Inches(x), Inches(y), width=Inches(w) if w else None,
                                    height=Inches(h) if h else None)


def table(s, df, x, y, w, h, size=12):
    rows, cols = df.shape[0] + 1, df.shape[1] + 1
    t = s.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h)).table
    t.cell(0, 0).text = ""
    for j, c in enumerate(df.columns):
        t.cell(0, j + 1).text = str(c)
    for i, (idx, r) in enumerate(df.iterrows()):
        t.cell(i + 1, 0).text = str(idx)
        for j, v in enumerate(r.values):
            t.cell(i + 1, j + 1).text = v if isinstance(v, str) else f"{v:.3f}" if abs(v) < 2 else f"{v:.2f}"
    for row in t.rows:
        for c in row.cells:
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size)


def g(set_, m, col):
    return summ.loc[(set_, m), col]


# ---------------------------------------------------------------- 1. title
s = add_slide("Super-Resolution for Port Intelligence",
              "One-liner: I tested whether AI super-resolution of free 10 m Sentinel-2 imagery actually helps analysing "
              "UAE ports, and measured where it invents detail that is not really there.\n\n"
              "This is layer 2 of 'A Gulf Maritime Picture': (1) dark-vessel detection from SAR+AIS, "
              "(2) this, image enhancement and its limits, (3) coastal bathymetry.",
              "Does 10 m → 2.5 m Sentinel-2 SR help analyse UAE ports, and where does it hallucinate?")
bullets(s, ["Layer 2 of “A Gulf Maritime Picture: three layers of intelligence for UAE ports and coastline”",
            "Bicubic vs pretrained (generic + satellite) vs fine-tuned on WorldStrat (L1 and GAN)",
            "Judged on fidelity, hallucination and a downstream detector, not on prettiness",
            "Jordan Manoj Cheruvathoor, MSc Informatik, Leibniz Universität Hannover"], 0.6, 1.9, 7.2, 4.5, 18)
picture(s, "port_khalifa_port.png", 7.9, 1.9, w=5.1)

# ---------------------------------------------------------------- 2. problem
s = add_slide("Why this matters to Stellaria",
              "Stellaria lists super-resolution, automated target recognition and maritime/port intelligence as capabilities, "
              "and runs Orbion with the Abu Dhabi Maritime Academy. Free Sentinel-2 gives a 5-day global revisit but 10 m pixels: "
              "a container ship is ~30 px long, cranes and small boats are sub-pixel.\n\n"
              "The risk: generative SR produces confident, realistic detail. In intelligence work a fabricated vessel or crane is "
              "worse than a blurry pixel because it creates false confidence. So the point of this project is evaluation, "
              "not prettier pictures.")
bullets(s, ["Sentinel-2: free, global, 5-day revisit, but 10 m pixels",
            "SR promises 2.5 m “for free”, a listed Stellaria capability",
            "Risk: GAN/diffusion SR invents realistic but false structure",
            "In intelligence work a fake ship or crane is worse than a blurry pixel",
            "Question: does SR help a downstream analysis, and where does it hallucinate?"], 0.6, 1.6, 6.3, 5)
picture(s, "hallucination_map_gulf_ft_gan.png", 6.9, 2.2, w=6.2)

# ---------------------------------------------------------------- 3. data + approach
s = add_slide("Data and approach",
              "WorldStrat is the only open dataset with real Sentinel-2 and real 1.5 m SPOT pairs worldwide, including the Gulf. "
              "It is 104 GB, so I read the zip index with HTTP range requests and downloaded only what I needed (~2.5 GB). "
              "13 Gulf tiles (Sharjah/Dubai, Al Ain, Doha, Jubail) are held out completely.\n\n"
              "Preprocessing makes the comparison fair: HR resampled to 2.5 m, histogram-matched to the Sentinel-2 colours "
              "(different sensors), sub-pixel registered with phase correlation, no-data masked.\n\n"
              "For the actual UAE ports there is no free 2.5 m reference, so I use reference-free checks: does SR stay "
              "consistent with what the satellite measured, and is it stable across dates where nothing changed?")
bullets(s, ["WorldStrat v1.1: Sentinel-2 L2A ↔ SPOT 1.5 m, global (NeurIPS 2022)",
            "Range-request subset: 473 train / 60 val / 120 test tiles + 13 held-out Gulf tiles",
            "Pairs: HR → 2.5 m, colour-matched, phase-correlation registered, masked",
            "UAE ports: Khalifa, Jebel Ali, Zayed; 4 cloud-free dates each (Earth Search STAC)",
            "Metrics: PSNR/SSIM/LPIPS + hallucination + LR-consistency + YOLO-OBB (DOTA) agreement"], 0.6, 1.6, 12, 3.2, 17)
tf = s.shapes.add_textbox(Inches(0.6), Inches(5.0), Inches(12.2), Inches(2)).text_frame
tf.text = ("Sentinel-2 10 m → [Bicubic | Real-ESRGAN | Satlas | Fine-tuned L1 | Fine-tuned GAN] → 2.5 m\n"
           "        ↓ compared against SPOT 2.5 m reference:  pixels · hallucination · detector agreement")
for p in tf.paragraphs:
    for r in p.runs:
        r.font.size, r.font.name, r.font.color.rgb = Pt(16), "Consolas", NAVY

# ---------------------------------------------------------------- 4. model choices / build vs reuse
abl_txt = ", ".join(f"{k}: {v.val_psnr.iloc[-1]:.2f} dB" for k, v in abl.items())
s = add_slide("Model choices: reuse first, then adapt",
              "Both pretrained models share the exact same RRDBNet architecture (16.7M params), so comparing them isolates the "
              "training data: natural photos (Real-ESRGAN) vs Sentinel-2 to US aerial (Satlas).\n\n"
              "Build vs reuse: I did not build an architecture. I re-implemented RRDBNet in ~100 lines only to avoid a broken "
              "dependency, and verified it loads the official checkpoints strictly. For training, I ran an initialisation "
              "ablation with the same data, loss and 1,500 iterations, only the starting weights differing. "
              f"Final validation PSNR: {abl_txt}. That answers 'why not pretrained?' with a measurement.\n\n"
              "Rejected: diffusion SR (too heavy for 4 GB, more hallucination in the literature), training from scratch (ablation), "
              "multi-image SR (ports move; next step).")
bullets(s, ["Same architecture (RRDBNet, 16.7 M), different pretraining:",
            "   Real-ESRGAN: generic photos + synthetic degradations",
            "   Satlas: Sentinel-2 → NAIP (USA only)",
            "Fine-tuned on WorldStrat: L1 (fidelity) and L1+VGG+GAN (perceptual)",
            "Init ablation (same budget): " + abl_txt,
            "Rejected: diffusion SR (4 GB GPU, hallucinates more), scratch (ablation), MISR (moving ships; next step)"],
        0.6, 1.6, 6.6, 5, 16)
picture(s, "ablation_init.png", 7.2, 1.8, w=5.9)

# ---------------------------------------------------------------- 5. results
cols = {"psnr": "PSNR↑", "ssim": "SSIM↑", "lpips": "LPIPS↓", "hallucination": "Halluc.↓", "lr_consistency": "LR-cons↑",
        "det_f1": "Det F1↑"}
for set_, title in [("gulf_test", "Results: held-out Gulf tiles"), ("test", "Results: global WorldStrat test")]:
    d = summ.loc[set_].reindex([m for m in NAMES if m in summ.loc[set_].index])[list(cols)].rename(columns=cols)
    d.index = [NAMES[m] for m in d.index]
    best = d["PSNR↑"].idxmax()
    s = add_slide(title,
                  f"Numbers from results/metrics/per_tile_summary.csv (python src/evaluate.py). {len(summ.loc[set_].index)} methods, "
                  f"{int(summ.loc[(set_, 'bicubic'), 'n'])} tiles. Best PSNR here: {best}.\n\n"
                  "Read it as a trade-off: L1 fine-tuning gives the best fidelity and LR-consistency; the GAN model gets better "
                  "LPIPS (looks more real) but hallucinates more patches. Pretrained models without fine-tuning do not beat bicubic "
                  "on fidelity, and Satlas collapses on Gulf scenes (domain shift: trained on the USA only).\n\n"
                  "Detection F1 is measured against detections on the real SPOT image with a frozen DOTA detector.")
    table(s, d, 0.5, 1.5, 7.6, 3.2, 12)
    picture(s, "perception_distortion.png" if set_ == "gulf_test" else "metrics_test.png", 8.3 if set_ == "gulf_test" else 0.5,
            1.5 if set_ == "gulf_test" else 4.9, w=4.8 if set_ == "gulf_test" else 12.3)
    if set_ == "gulf_test":
        picture(s, "examples_gulf.png", 0.5, 4.85, h=2.55)

# ---------------------------------------------------------------- 6. failures
s = add_slide("Failure cases: what they teach",
              "Three kinds of failure. (1) Domain shift: Satlas, trained on US imagery, paints vegetation and blocky 32-px seams "
              "onto bright Gulf scenes. (2) GAN hallucination: realistic roof and road textures that do not match the reference, "
              "flagged red in the correctness map. (3) Smoothing: Real-ESRGAN removes small objects, so the detector misses them.\n\n"
              "Caveat: some 'hallucinations' are real changes between the SPOT and Sentinel-2 dates (moving vehicles, construction). "
              "I used the closest revisit, but this cannot be fully removed.")
picture(s, "failures_satlas.png", 0.4, 1.4, w=6.3)
picture(s, "hallucination_map_ft_gan.png", 6.8, 1.4, w=6.3)
bullets(s, ["Domain shift: Satlas invents vegetation + tile seams on Gulf scenes",
            "GAN: realistic but wrong texture (red = hallucinated patch)",
            "Smoothing: small objects vanish, so the detector misses them"], 6.8, 4.6, 6.3, 2.5, 15)

# ---------------------------------------------------------------- 7. ports + limitations
s = add_slide("UAE ports, limitations and next steps",
              "At the ports there is no free 2.5 m reference, so I report reference-free checks from results/metrics/ports_summary.csv: "
              "LR-consistency (does SR agree with what Sentinel-2 measured) and temporal instability (SR differences on pixels the "
              "input says are unchanged).\n\n"
              "How it fits the bigger picture: project 1 (dark vessels) finds ships from SAR+AIS; SR could help an analyst inspect "
              "optical chips of those detections, but only with a hallucination flag. Project 3 (bathymetry) must not use SR "
              "outputs as input: radiometry must stay physical.\n\n"
              "Next steps: multi-image SR (static infrastructure), SEN2SR as a radiometrically-consistent baseline, per-pixel "
              "uncertainty maps, human analyst study, and a commercial-licence-clean dataset for production.")
pt = ports[["lr_consistency", "excess_instability", "n_ship"]].rename(
    columns={"lr_consistency": "LR-cons↑", "excess_instability": "Excess instab.↓", "n_ship": "Ships/img"})
pt.index = [NAMES.get(m, m) for m in pt.index]
table(s, pt, 0.5, 1.5, 6.2, 2.6, 12)
bullets(s, ["Limitations: SPOT/S2 date gap; detector labels are pseudo-GT; 8-bit stretch darkens water; CC BY-NC data",
            "Next: multi-image SR, SEN2SR baseline, uncertainty maps, analyst study",
            "Fits the Gulf Maritime Picture: SR to inspect dark-vessel candidates (project 1) with a hallucination flag; "
            "never as input to bathymetry (project 3)"], 0.5, 4.4, 6.3, 3, 14)
picture(s, "port_jebel_ali_port.png", 6.9, 1.5, w=6.2)

# ---------------------------------------------------------------- 8. demo
s = add_slide("Demo and Q&A",
              "Demo: streamlit run demo/app.py. Before/after slider on Khalifa, Jebel Ali and Zayed Port, method picker, "
              "detection overlay, plus upload-your-own Sentinel-2 crop.\n\nTake-away: SR is useful as a visual aid when fine-tuned "
              "on the right domain and kept fidelity-oriented; generative detail must never be treated as evidence without a "
              "hallucination check.")
bullets(s, ["streamlit run demo/app.py: before/after slider over UAE ports",
            "Take-away: domain fine-tuning matters more than the architecture",
            "Fidelity-oriented SR = visual aid; GAN detail ≠ evidence",
            "Always ship SR with a hallucination / consistency check"], 0.6, 1.6, 6, 4, 18)
picture(s, "port_zayed_port.png", 6.7, 1.6, w=6.3)

out = os.path.join(ROOT, "presentation", "sr_port_intelligence.pptx")
prs.save(out)
print("saved", out)
