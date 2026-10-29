# RPTM: A Geometric Prototype Matching Framework for Incomplete Multi-View Clustering

This repository provides an implementation of **RPTM**, a two-stage framework for incomplete multi-view clustering. RPTM combines geometry-conditioned representation correction with Tangent–Cosine prototype matching to address incomplete observations and improve clustering in a shared representation space.

The implementation includes the complete training and clustering pipeline, dataset preprocessing, missing-view generation, evaluation, and experiment configuration.

## Overview

Incomplete multi-view clustering aims to group samples when some views are unavailable. RPTM addresses this problem through two complementary stages.

**Stage 1: Pullback-Conditioned Representation Correction**

The first stage learns a shared representation and corrects incomplete-view representations using decoder-induced Pullback geometry. The training objective incorporates reconstruction, correction, and neighborhood-preservation terms.

**Stage 2: Tangent–Cosine Prototype Matching**

The second stage performs clustering using a matching cost that combines cosine distance with a geometry-conditioned tangent component. It alternates between sample assignment and prototype optimization under the unit-sphere constraint.

The implementation provides a unified pipeline for training, clustering, and evaluation across multiple incomplete multi-view datasets.

## Repository Structure

```text
.
├── model.py
├── train.py
├── data.py
├── clustering.py
├── train_mvc.py
├── requirements.txt
├── README.md
└── .gitignore
```

- `model.py`: Shared representation learning, decoder-based geometric correction, and training losses.
- `train.py`: First-stage model training.
- `data.py`: Dataset loading, preprocessing, and missing-view mask generation.
- `clustering.py`: Tangent–Cosine matching, constrained sample assignment, and prototype optimization.
- `train_mvc.py`: Unified experiment entry point, dataset configurations, and result generation.
- `requirements.txt`: Python dependencies.

## Installation

Python 3.10 or later and a compatible PyTorch installation are recommended.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Install a PyTorch build compatible with your hardware and CUDA environment if GPU acceleration is required.

## Supported Benchmarks

The implementation supports the following five multi-view clustering benchmarks:

| Dataset | Number of views | Number of clusters |
|---|---:|---:|
| Reuters | 5 | 6 |
| ALOI | 4 | 100 |
| COIL20 | 3 | 20 |
| MSRC_v1 | 5 | 7 |
| Mfeat | 6 | 10 |

These datasets serve as evaluation benchmarks for the same overall RPTM pipeline. Dataset-specific configurations specify the experimental settings used for each benchmark; they do not define separate clustering algorithms.

## Data Preparation

Dataset files are not included in this repository. Place the corresponding MATLAB files in the `data/` directory or provide their paths through `--data-file`.

The default filenames and supported data formats are listed below.

| Dataset | Default filename | Supported data fields |
|---|---|---|
| Reuters | `Reuters.mat` | `X` (view cell array), `y` |
| ALOI | `ALOI.mat` | `X` (view cell array), `y` |
| COIL20 | `COIL20.mat` | `X` (view cell array), `y` |
| MSRC_v1 | `MSRC_v1.mat` | `msr1`–`msr5`, `X1`–`X5`, or a five-view `X` cell array; labels are also required |
| Mfeat | `Mfeat.mat` | `X` (view cell array), `Y` |

The loader should be configured according to the actual field names and array layout of each dataset file.

Labels are used for evaluation and are not intended to participate in representation learning, prototype initialization, or clustering optimization.

## Experimental Configuration

The same two-stage algorithm is used across all supported benchmarks. Dataset-specific hyperparameters are defined in `train_mvc.py` to accommodate differences in dataset scale, feature dimensionality, and experimental settings.

The following table summarizes the main configuration values.

| Dataset | Epochs | Latent dimension | Encoder hidden layers | Recovery hidden dimension | Ridge | `lambda_corr` | `lambda_nbr` | `tau_n` | Preprocessing |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| Reuters | 20 | 64 | 512 | 512 | 0.005 | 1.0 | 0.3 | 0.2 | TF-IDF, SVD-100, L2 normalization |
| ALOI | 80 | 128 | 256, 128 | 128 | 0.005 | 0.1 | 1.0 | 0.2 | Z-score |
| COIL20 | 200 | 64 | 512 | 512 | 0.005 | 1.0 | 0.3 | 0.2 | Z-score |
| MSRC_v1 | 300 | 64 | 512 | 512 | 0.002 | 1.0 | 0.1 | 0.2 | Z-score |
| Mfeat | 80 | 64 | 512 | 512 | 0.005 | 1.0 | 0.3 | 0.2 | Z-score |

Other parameters, including optimizer settings, Pullback computation settings, neighborhood configuration, and prototype optimization parameters, are defined in the configuration and implementation files.

Unless explicitly overridden, the configured values remain fixed across missing rates and random seeds for each dataset.

## Running Experiments

The unified entry point is `train_mvc.py`.

For example, to evaluate RPTM on COIL20 under several missing rates:

```bash
python train_mvc.py \
    --dataset coil20 \
    --data-file data/COIL20.mat \
    --mr 0.1 0.3 0.5 0.7 \
    --seeds 0 1 2 3 4
```

To run the other supported benchmarks:

```bash
python train_mvc.py --dataset reuters --data-file data/Reuters.mat
python train_mvc.py --dataset aloi --data-file data/ALOI.mat
python train_mvc.py --dataset msrc_v1 --data-file data/MSRC_v1.mat
python train_mvc.py --dataset mfeat --data-file data/Mfeat.mat
```

The device defaults to CUDA when available and otherwise falls back to CPU. To specify the device explicitly, use:

```bash
--device cuda:0
```

or:

```bash
--device cpu
```

Additional command-line options allow selected training and clustering parameters to be overridden. Run the following command to inspect the options supported by the current implementation:

```bash
python train_mvc.py --help
```

## Evaluation and Outputs

The implementation evaluates clustering performance using standard metrics, including:

- Clustering accuracy (ACC)
- Normalized mutual information (NMI)
- Adjusted Rand index (ARI)

Experiment outputs include per-run configurations and metrics, predicted cluster assignments, and aggregated summaries across completed runs.

Ground-truth labels are used for evaluation only. They should not be used to select hyperparameters, choose checkpoints, or select clustering solutions based on evaluation performance.

## Reproducibility

For reproducible experiments, record the dataset version, preprocessing configuration, missing rate, random seeds, model parameters, and software environment.

The model initialization seed and missing-mask seed control different sources of randomness. For fair comparisons, use consistent missing masks when comparing methods or configurations.

Dataset-specific hyperparameters should be reported transparently, and all experimental comparisons should follow a consistent evaluation protocol.

## Scope and Limitations

This repository provides the source implementation and experiment pipeline for RPTM. The included code should be treated as an implementation release until the reported results have been independently reproduced on the original datasets.

The availability of source code and successful synthetic-data tests does not by itself establish reproduction of the numerical results reported in the paper.

Please consult the source code and experiment configurations for implementation details and limitations.
