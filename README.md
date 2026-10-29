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
| Reuters | 20 | 64 | 512 | 512 | 0.005 | 1.0 | 0.3 | 0.2 | TF-IDF + SVD-100 + L2 | cached |
| ALOI | 80 | 128 | 256, 128 | 128 | 0.005 | 0.1 | 1.0 | 0.2 | z-score | original |
| COIL20 | **200** | **64** | **512** | **512** | **0.005** | **1.0** | **0.3** | **0.2** | z-score | original |
| MSRC_v1 | 300 | 64 | 512 | 512 | 0.002 | 1.0 | 0.1 | 0.2 | z-score | original |
| Mfeat | 80 | 64 | 512 | 512 | 0.005 | 1.0 | 0.3 | 0.2 | z-score | original |

The **COIL20** entry explicitly specifies every active `Config` setting so its parameters can be inspected without relying on implicit shared defaults:

| Parameter | Value | Parameter | Value |
|---|---:|---|---:|
| `clusters` | 20 | `latent_dim` | 64 |
| `epochs` | 200 | `hidden` | (512,) |
| `recovery_hidden` | 512 | `lr` | 0.001 |
| `batch_size` | 256 | `preprocess` | zscore |
| `alpha_g` | 1.0 | `ridge` | 0.005 |
| `mu_min` | 1e-8 | `trace_probes` | 8 |
| `r_max` | pi/2 | `eps_w` | 1e-8 |
| `eps_ang` | 1e-6 | `eps_f` | 1e-8 |
| `cg_max_iter` | 64 | `cg_rtol` | 1e-5 |
| `cg_atol` | 1e-9 | `lambda_corr` | 1.0 |
| `lambda_nbr` | 0.3 | `tau_n` | 0.2 |
| `rho_cluster` | 0.4 | `tau_cluster` | 0.1 |
| `teacher_kmeans_iters` | 20 | `teacher_kmeans_restarts` | 3 |
| `lambda_cos` | 0.7 | `lambda_tan` | 0.3 |
| `beta` | 16.0 | `cluster_max_iter` | 100 |
| `cluster_tol` | 1e-6 | `backtracking_steps` | 30 |
| `backtracking_factor` | 0.5 | `match_batch_size` | 1024 |
| `n_streams` | 8 | `selection_tol` | 0.02 |
| `pullback_backend` | original | | |

The COIL20 configuration was transcribed from the supplied `train_mvc.py` implementation. The separately supplied `run8(3).py` and `run_one.py` scripts concern an **ALOI-only `c04_teacher` experiment**, not COIL20, and are not included in this repository.

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
