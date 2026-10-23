"""Offline orchestrator for the whole training + evaluation pipeline (runs without internet, strictly sequential).

0. init ablation: L1 fine-tune from realesrgan / satlas / scratch, same budget (skipped if already finished)
1. pick the best initialisation from the ablation logs (best val PSNR after training started)
2. fine-tune L1 model from it          -> weights/finetuned_l1.pth
3. GAN fine-tune from the L1 model      -> weights/finetuned_gan.pth
4. evaluate all methods (WorldStrat test + Gulf), port evaluation, figures, slide deck

Usage:  python src/run_all.py
"""
import json
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(args):
    print("\n>>>", " ".join(args), flush=True)
    subprocess.run([PY, "-u", "-W", "ignore"] + args, cwd=ROOT, check=True)


def finished(name, iters):
    p = os.path.join(ROOT, "results", "training", name, "log.json")
    return os.path.exists(p) and json.load(open(p))[-1]["it"] == iters


def main():
    for k in ["realesrgan", "satlas", "scratch"]:
        if not finished(f"ablation_l1_{k}", 1500):
            run(["src/train.py", f"configs/ablation_l1_{k}.yaml"])
    scores = {}
    for k in ["realesrgan", "satlas", "scratch"]:
        p = os.path.join(ROOT, "results", "training", f"ablation_l1_{k}", "log.json")
        if os.path.exists(p):
            log = json.load(open(p))
            scores[k] = max(r["val_psnr"] for r in log if r["it"] > 0)
    best = max(scores, key=scores.get)
    print("ablation best val PSNR per init:", scores, "-> best init:", best, flush=True)
    json.dump({"scores": scores, "best_init": best}, open(os.path.join(ROOT, "results", "metrics", "ablation_choice.json"), "w"), indent=1)

    cfg = dict(name="ft_l1", init=best, mode="l1", iters=4000, batch=8, lr_patch=32, lr_g=1e-4, val_every=500,
               n_val=60, seed=0, save_to="weights/finetuned_l1.pth")
    with open(os.path.join(ROOT, "configs", "ft_l1.yaml"), "w") as f:
        f.write(f"# L1 fine-tune from the best ablation init ({best}), longer budget, all 60 val tiles for selection.\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    if not os.path.exists(os.path.join(ROOT, "weights", "finetuned_l1.pth")):
        run(["src/train.py", "configs/ft_l1.yaml"])
    if not os.path.exists(os.path.join(ROOT, "weights", "finetuned_gan.pth")):
        run(["src/train.py", "configs/ft_gan.yaml"])
    run(["src/evaluate.py", "--methods", "bicubic", "realesrgan", "satlas", "ft_l1", "ft_gan", "--save-sr"])
    run(["src/eval_ports.py", "--methods", "bicubic", "realesrgan", "satlas", "ft_l1", "ft_gan"])
    run(["src/make_figures.py"])
    run(["presentation/build_deck.py"])
    print("\nPIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
