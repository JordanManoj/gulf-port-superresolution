"""Streamlit demo: before/after slider of Sentinel-2 super-resolution over UAE ports.

Run:  streamlit run demo/app.py
Needs results/ports/... (python src/eval_ports.py) and results/metrics/*.csv (python src/evaluate.py).
Optional: upload your own Sentinel-2 RGB PNG (10 m) and super-resolve it live on GPU/CPU.
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

st.set_page_config(page_title="SR for Port Intelligence", layout="wide")
LABEL = {"bicubic": "Bicubic (baseline)", "realesrgan": "Real-ESRGAN (pretrained)", "satlas": "Satlas (pretrained, satellite)",
         "ft_l1": "Fine-tuned on WorldStrat (L1)", "ft_gan": "Fine-tuned on WorldStrat (GAN)"}
PORTS = os.path.join(ROOT, "results", "ports")
MET = os.path.join(ROOT, "results", "metrics")


def rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


st.title("Super-Resolution for Port Intelligence: does it help, and where does it hallucinate?")
st.caption("Sentinel-2 L2A 10 m → 2.5 m (×4). Methods compared on WorldStrat (real SPOT 1.5 m references) and on UAE ports.")

tab1, tab2, tab3 = st.tabs(["UAE ports: before / after", "Results on WorldStrat", "Try your own image"])

with tab1:
    ports = sorted(os.listdir(PORTS)) if os.path.isdir(PORTS) else []
    if not ports:
        st.warning("Run `python src/eval_ports.py` first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        port = c1.selectbox("Port", ports, format_func=lambda p: p.replace("_", " ").title())
        date = c2.selectbox("Sentinel-2 date", sorted(os.listdir(os.path.join(PORTS, port))))
        avail = [m for m in LABEL if os.path.exists(os.path.join(PORTS, port, date, f"{m}.png"))]
        left = c3.selectbox("Left", avail, index=0, format_func=LABEL.get)
        right = c4.selectbox("Right", avail, index=len(avail) - 1, format_func=LABEL.get)
        show_det = st.checkbox("Show YOLO11-OBB (DOTA) detections", value=False)
        ims = {m: rgb(os.path.join(PORTS, port, date, f"{m}.png")) for m in (left, right)}
        if show_det:
            det = json.load(open(os.path.join(MET, "ports_detections.json")))
            for m in ims:
                im = ims[m].copy()
                for poly, cl in zip(det[f"{port}/{date}/{m}"]["boxes"], det[f"{port}/{date}/{m}"]["classes"]):
                    cv2.polylines(im, [np.array(poly, np.int32)], True, (0, 255, 255) if cl == "ship" else (255, 255, 0), 2)
                ims[m] = im
        try:
            from streamlit_image_comparison import image_comparison
            image_comparison(img1=ims[left], img2=ims[right], label1=LABEL[left], label2=LABEL[right], width=1100)
        except ImportError:  # started outside the project venv: show side by side instead of a slider
            st.info("Slider unavailable in this Python environment. Start the demo with run_demo.ps1 "
                    "(or .venv\\Scripts\\python -m streamlit run demo/app.py).")
            ca, cb = st.columns(2)
            ca.image(ims[left], caption=LABEL[left])
            cb.image(ims[right], caption=LABEL[right])
        p = os.path.join(MET, "ports_per_image.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            st.dataframe(df[(df.port == port) & (df.date == date)].set_index("method").round(2))

with tab2:
    s = os.path.join(MET, "per_tile_summary.csv")
    if os.path.exists(s):
        summ = pd.read_csv(s)
        st.dataframe(summ.set_index(["set", "method"])[["psnr", "ssim", "lpips", "edge_f1", "improvement", "hallucination",
                                                         "lr_consistency", "det_f1"]].round(3))
    for f in ["metrics_test.png", "metrics_gulf_test.png", "perception_distortion.png", "hallucination_map_gulf_ft_gan.png",
              "failures_satlas.png", "ablation_init.png"]:
        p = os.path.join(ROOT, "results", "figures", f)
        if os.path.exists(p):
            st.image(p)

with tab3:
    up = st.file_uploader("Sentinel-2 true-colour PNG/JPG at 10 m (small crops, < 400×400 px)", type=["png", "jpg", "jpeg"])
    method = st.selectbox("Model", [m for m in LABEL if m != "bicubic"], format_func=LABEL.get)
    if up is not None:
        import torch
        from srlib import WEIGHTS, bicubic, load_model, sr_tiled
        lr = cv2.cvtColor(cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)[:400, :400]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if not os.path.exists(os.path.join(ROOT, WEIGHTS[method])):
            st.error(f"weights missing: {WEIGHTS[method]}")
        else:
            with st.spinner("Super-resolving…"):
                sr = sr_tiled(load_model(method, dev), lr, device=dev)
            from streamlit_image_comparison import image_comparison
            image_comparison(img1=bicubic(lr), img2=sr, label1="Bicubic", label2=LABEL[method], width=1000)
