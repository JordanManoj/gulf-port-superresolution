# Data

Nothing in `data/` is committed. Everything is re-created by scripts.

| Folder | Content | How to get it | Size |
|---|---|---|---|
| `data/worldstrat/` | WorldStrat v1.1 subset: HR SPOT RGB (`hr/`), Sentinel-2 L2A revisits (`lr/`), `metadata.csv`, official split, `subset.csv` | `python src/download_worldstrat.py` | ~2.5 GB |
| `data/pairs/` | Aligned, cached training/eval pairs (`.npz`: LR uint8, HR uint8 colour-matched + registered, valid mask) | `python src/prepare_pairs.py` | ~1 GB |
| `data/s2_ports/` | Sentinel-2 L2A RGB crops (10 m, native UTM) over Khalifa Port, Jebel Ali, Zayed Port, 4 cloud-free dates each | `python src/fetch_s2_ports.py` | ~30 MB |

## WorldStrat (paired Sentinel-2 / SPOT)
- Paper: Cornebise, Oršolić, Kalaitzis, NeurIPS 2022 Datasets & Benchmarks, https://arxiv.org/abs/2207.06418
- Zenodo v1.1: https://zenodo.org/records/15382551 (~104 GB of zips)
- Licence: SPOT HR imagery **CC BY-NC 4.0** (non-commercial), Sentinel-2 + labels CC BY 4.0
- We never download the full archives: `src/remote_zip.py` reads the zip index via HTTP range requests and fetches single files.
- Download first: `metadata.csv` and `stratified_train_val_test_split.csv` (small, plain downloads), then run the script:
  ```
  curl -L -o data/worldstrat/metadata.csv "https://zenodo.org/records/15382551/files/metadata.csv?download=1"
  curl -L -o data/worldstrat/stratified_train_val_test_split.csv "https://zenodo.org/records/15382551/files/stratified_train_val_test_split.csv?download=1"
  python src/download_worldstrat.py --n-train 600 --n-val 60 --n-test 120
  ```

### Subset
| Set | Tiles | LR revisits / tile | Selection |
|---|---|---|---|
| train | 600 | up to 3 | official `train` split, 50 % Settlement |
| val | 60 | 1 | official `val` split, 50 % Settlement |
| test | 120 | 1 | official `test` split, 50 % Settlement |
| gulf_test | 13 | 1 | **every** tile in lat 22–30N, lon 47–58E, any split, never trained on |

Revisits are the lowest-cloud (≤5 %) ones closest in time to the SPOT acquisition.

## Sentinel-2 over UAE ports
- Source: Element 84 Earth Search STAC (`https://earth-search.aws.element84.com/v1`, collection `sentinel-2-l2a`), AWS Open Data, no account needed.
- AOIs: `aoi/*.geojson` (6×6 km boxes).
- Reflectance = DN × 0.0001 − 0.1 (processing baseline ≥ 04.00), taken from the STAC `raster:bands` metadata.

## Pretrained weights (`weights/`, not committed)
```
curl -L -o weights/RealESRGAN_x4plus.pth      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
curl -L -o weights/RealESRGAN_x4plus_netD.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth
curl -L -o weights/esrgan_1S2.pth             https://storage.googleapis.com/satlas-satellite-super-resolution/esrgan_1S2.pth
```
YOLO11m-OBB (DOTA) is downloaded automatically by Ultralytics (`yolo11m-obb.pt`); move it to `weights/`.
