# Reproducing ISBI 2022 (RU-VAE) experiments

This repository is trimmed to the **malaria patch** and **nanofibre** setups used in thesis Chapter 4. Paths in legacy shell scripts may still point to old cluster locations; override them with your local dataset roots.

## Environment

```bash
cd src
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

Use a PyTorch build compatible with your CUDA driver if training on GPU.

## Datasets

- **Malaria:** point `--data_path` at the dataset root that contains either prebuilt `cached/*.pt` tensors or the raw layout described in `src/datasets/malaria.py` (see the WACV 2022 / ss-gVAE repo for the same extraction protocol).
- **Nanofibre:** ImageFolder layout with `train/` and `test/` subdirectories under `--data_path` (see `src/datasets/nanofibre.py`).

## Deep RU-VAE / DeepSAD-style training

From `src/` (so `PYTHONPATH` includes the package roots as in the original layout):

```bash
python main.py malaria_dataset cifar10_LeNet <xp_dir> <data_path> [options...]
python main.py nanofibre nanofibre_vae <xp_dir> <data_path> [options...]
```

Create `<xp_dir>` before running (Click requires it to exist).

## Hybrid SSAD baseline

```bash
python baseline_ssad.py malaria_dataset <xp_dir> <data_path> --hybrid True --load_ae /path/to/model.tar ...
```

`--load_ae` must match the architecture saved in `model.tar` (`malaria_net` vs `nanofibre_vae`).

## Binary classifier baseline

See `binary_classifier/` and `binary_classifier/baseline_binary_classifier.py`.
