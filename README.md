# RPTM: Incomplete Multi-View Clustering

This repository contains a two-stage implementation of RPTM for **Reuters, ALOI, COIL20, MSRC_v1, and Mfeat**. It includes Pullback-conditioned representation correction and Tangent–Cosine prototype matching.

## Installation

Python 3.10+ and a compatible PyTorch build are required. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data format

Dataset files are not distributed in this repository. Put files in `data/` or supply `--data-file`.

| Dataset | Filename | Classes | Required MAT fields |
|---|---|---:|---|
| Reuters | `Reuters.mat` | 6 | `X` (view cell array), `y` |
| ALOI | `ALOI.mat` | 100 | `X` (view cell array), `y` |
| COIL20 | `COIL20.mat` | 20 | `X` (view cell array), `y` |
| MSRC_v1 | `MSRC_v1.mat` | 7 | `msr1`...`msr5` or `X1`...`X5`, plus labels; 5-view `X` cell arrays are supported |
| Mfeat | `Mfeat.mat` | 10 | `X` (view cell array), `Y` |

Labels are used only for evaluation. Observed feature vectors are preprocessed independently in each view. Reuters uses TF-IDF and truncated SVD (100 components), followed by L2 normalization.

## Dataset configurations

All values below are loaded from `DATASETS` and `Config` in `train_mvc.py`. Parameters are constant across missing rates and seeds unless overridden explicitly.

| Dataset | Epochs | Latent dim. | Hidden layers | Recovery hidden | Ridge | `lambda_corr` | `lambda_nbr` | `tau_n` | Preprocessing | Pullback backend |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| Reuters | 20 | 64 | 512 | 512 | 0.005 | 1.0 | 0.3 | 0.2 | TF-IDF + SVD-100 + L2 | original |
| ALOI | 80 | 128 | 256, 128 | 128 | 0.005 | 0.1 | 1.0 | 0.2 | z-score | original |
| COIL20 | **200** | **64** | **512** | **512** | **0.005** | **1.0** | **0.3** | **0.2** | z-score | original |
| MSRC_v1 | 300 | 64 | 512 | 512 | 0.002 | 1.0 | 0.1 | 0.2 | z-score | original |
| Mfeat | 80 | 64 | 512 | 512 | 0.005 | 1.0 | 0.3 | 0.2 | z-score | original |

## Usage

Run COIL20 with its declared defaults:

```bash
python train_mvc.py --dataset coil20 --data-file data/COIL20.mat --mr 0.1 0.3 0.5 0.7 --seeds 0 1 2 3 4
```

Run other datasets:

```bash
python train_mvc.py --dataset reuters --data-file data/Reuters.mat
python train_mvc.py --dataset aloi --data-file data/ALOI.mat
python train_mvc.py --dataset msrc_v1 --data-file data/MSRC_v1.mat
python train_mvc.py --dataset mfeat --data-file data/Mfeat.mat
```

The default device is CUDA if available and CPU otherwise. `--device cuda:0` and `--device cpu` explicitly select a device. Options include `--epochs`, `--latent`, `--hidden`, `--lambda-corr`, `--lambda-nbr`, `--rho-cluster`, `--streams`, and `--selection-tol`. The model initialization seed defaults to zero (`--model-seed`), while `--seeds` varies the incomplete-view masks. These are distinct controls.

## Results

The experiment directory contains per-run `config.json`, `metrics.json`, and `labels.npy`. `summary.csv` and `summary.json` contain means and standard deviations over completed runs. Ground-truth labels are not used in optimization or restart selection.

## Scope and limitations

This is a source-cleaned release candidate, not an independently verified reproduction of the numerical results on the five original datasets. The included source and synthetic-data runtime checks do not establish real-data clustering accuracy. Choose an appropriate license before publishing the repository.
