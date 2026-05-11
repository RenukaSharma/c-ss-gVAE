# RU-VAE (ISBI 2022) — thesis Chapter 4

Code for **robust semi-supervised VAE** experiments on **malaria blood-smear patches** and **nanofibre** imagery, as in the ISBI 2022 paper and PhD thesis Chapter 4.

> This repository is **not** the ss-gVAE (WACV 2022) codebase. For ss-gVAE, see [`RenukaSharma/ss-gVAE`](https://github.com/RenukaSharma/ss-gVAE). A compact map of thesis chapters to repositories is maintained in the ss-gVAE README.

## Paper and camera-ready PDF

- **IEEE Xplore (ISBI 2022):** [document 9761472](https://ieeexplore.ieee.org/document/9761472)  
- **Camera-ready PDF (Google Drive):** [link](https://drive.google.com/file/d/1fapKVmd193qVkeJokayVqOOKCsm52Kqg/view)

## Scope of this release

The tree is reduced to **two datasets** and the networks used with them:

| Dataset (CLI name) | Typical `net_name` |
|--------------------|--------------------|
| `malaria_dataset`  | `cifar10_LeNet` (32×32 RGB patches) or `malaria_net` |
| `nanofibre`        | `nanofibre_vae` |

Supporting code lives under `src/` (DeepSAD-style training, SSAD hybrid baseline) and `binary_classifier/` (binary-classifier baseline).

## Quick start

See [`docs/reproducing_paper.md`](docs/reproducing_paper.md) for environment setup, data layout, and example commands.

## Citation

See [`CITATION.cff`](CITATION.cff) for software metadata and the **preferred citation** to the ISBI paper. If you cite the repository software separately, credit the maintainer listed there.

## License

[MIT](LICENSE). Copyright line may be adjusted if your institution requires a different attribution; open an issue or PR text if needed.
