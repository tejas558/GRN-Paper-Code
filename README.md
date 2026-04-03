# Causal Dynamic GRN — Delay-Aware Neural ODE for Time-Series Single-Cell Multiome

A delay-aware Neural ODE model that infers **time-varying gene regulatory networks (GRNs)** from paired scATAC + scRNA (10x Multiome) time-series data.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Input Data Requirements](#input-data-requirements)
- [Configuration Reference](#configuration-reference)
- [Running the Pipeline](#running-the-pipeline)
- [Generating Evaluation Figures](#generating-evaluation-figures)
- [Running Everything at Once](#running-everything-at-once)
- [Understanding the Outputs](#understanding-the-outputs)
- [In Silico TF Perturbation](#in-silico-tf-perturbation)
- [Example Results (GSE318326)](#example-results-gse318326)
- [Repository Structure](#repository-structure)

---

## Overview

This model takes 10x Multiome H5 files across multiple time points and conditions, computes pseudotime trajectories per condition, bins cells, and trains a coupled delay-ODE system to learn:

- **Time-varying TF→target edge weights** — the dynamic GRN
- **TF activity gates** — how open chromatin gates transcription
- **Regulatory delays** — learned lag between chromatin accessibility and gene expression

Trained on GSE318326 (human plasma cell differentiation, DMSO vs A366 treatment), the model achieves **R² > 0.99** reconstruction on TF RNA and target RNA trajectories.

---

## Model Architecture

For each pseudotime bin `t`, the model tracks four coupled state blocks:

| Block | Symbol | Description |
|-------|--------|-------------|
| TF accessibility | `a(t)` | ATAC promoter accessibility for each TF |
| TF RNA | `r(t)` | TF mRNA levels |
| Target RNA | `x(t)` | Downstream target gene expression |
| Latent state | `z(t)` | Hidden regulatory context |

The dynamics are governed by:

```
da/dt = f_a( a(t−τ₁), r(t), z(t), t )
dr/dt = f_r( a(t−τ₁), r(t), z(t), t )
dx/dt = basal(x(t), z(t), t) + W(t,z) @ gate( a(t−τ₂), r(t−τ₂), z(t), t )
dz/dt = f_z( a(t), r(t), x(t), z(t), t )
```

- `W(t, z)` is the **dynamic GRN matrix** (TFs × targets), factored as a low-rank product conditioned on time and latent state
- `τ₁`, `τ₂` are **trainable delays** (accessibility→TF RNA, TF state→target RNA)
- The ODE is integrated with a fixed-step differentiable delay integrator

---

## Installation

```bash
cd GRN-Github
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Alternatively, without installing as a package:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- numpy, pandas, scipy, scikit-learn
- h5py (reading 10x H5 files)
- pyranges (GTF parsing and promoter overlap)
- matplotlib, seaborn (evaluation figures)
- pyyaml

---

## Quick Start

```bash
# 1. Copy and edit the config with your file paths
cp configs/example_project.yaml configs/my_project.yaml
# Edit configs/my_project.yaml — set gtf, output_dir, and sample paths

# 2. Train the model
python run_pipeline.py --config configs/my_project.yaml --project gse318326_human_plasma

# 3. Evaluate and generate figures
python scripts/evaluate_grn_model.py \
  --run-dir outputs/gse318326_human_plasma \
  --results-dir results/gse318326_eval
```

---

## Input Data Requirements

### 1. 10x Multiome H5 files

Per-timepoint `filtered_feature_bc_matrix.h5` files from 10x Genomics Multiome (joint ATAC + RNA). Tested with:

- **GSE318326** — human plasma cell differentiation, 8 time-point files across 2 conditions (DMSO, A366)
- **GSE223041** — mouse mESC, either per-sample H5s or an aggregated H5 + barcode metadata CSV

### 2. GTF annotation

A genome annotation GTF (`.gtf` or `.gtf.gz`) for the correct species/build. Used to map ATAC peaks to promoter windows.

- Human: GENCODE v46 (`GRCh38.genes_only.gtf.gz`)
- Mouse: GENCODE vM35

### 3. TF list (optional but recommended)

A plain text file, one gene symbol per line. Without this, the model falls back to highly variable genes as candidate TFs — biologically less meaningful.

### GSE223041 metadata (if using aggregated H5)

```bash
python scripts/build_gse223041_metadata_template.py \
  --h5 /path/to/GSE223041_Aggregated_filtered_feature_bc_matrix.h5 \
  --out /path/to/gse223041_cell_metadata.csv
```

Fill in `time`, `replicate`, and `condition` columns before running the pipeline.

---

## Configuration Reference

The YAML config file has one entry per project under the `projects:` key. See `configs/example_project.yaml` for a full template.

### Top-level fields

| Field | Description |
|-------|-------------|
| `name` | Project identifier, also used as output subfolder name |
| `species` | `human` or `mouse` |
| `gtf` | Path to GTF annotation file |
| `tf_list` | (Optional) path to TF gene symbol list |
| `output_dir` | Root directory for model outputs |

### `preprocessing` block

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_tfs` | 100 | Number of TF features to select |
| `n_targets` | 400 | Number of target genes to select |
| `n_bins` | 32 | Pseudotime bins per trajectory |
| `min_cells_per_bin` | 20 | Minimum cells required per bin |
| `promoter_upstream` | 2000 | bp upstream of TSS for promoter window |
| `promoter_downstream` | 500 | bp downstream of TSS for promoter window |

### `model` block

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 64 | Dimension of latent regulatory state `z` |
| `hidden_dim` | 256 | Hidden units in ODE drift networks |
| `edge_rank` | 16 | Low-rank factorization rank for GRN matrix `W` |
| `epochs` | 800 | Maximum training epochs |
| `lr` | 0.001 | Adam learning rate |
| `max_tau` | 0.35 | Maximum trainable delay (trajectory units) |
| `max_step` | 0.02 | Fixed ODE integration step size |
| `early_stopping_patience` | 60 | Epochs without improvement before stopping |

### `samples` block

Each entry is one H5 file:

```yaml
samples:
  - path: /path/to/file.h5
    time: 4          # numeric time label (e.g. day 4)
    condition: DMSO  # condition name
    replicate: rep1  # replicate label
```

---

## Running the Pipeline

```bash
python run_pipeline.py \
  --config configs/gse318326_run.yaml \
  --project gse318326_human_plasma
```

Optional arguments:

| Argument | Description |
|----------|-------------|
| `--output-dir PATH` | Override the output directory from config |
| `--perturb-tf GENE` | Run in silico TF knockout after training (repeatable) |
| `--perturb-start FLOAT` | Knockout start time in [0, 1] normalized trajectory units (default: 0.5) |
| `--perturb-mode {acc,rna,both}` | Zero TF accessibility, TF RNA, or both (default: both) |

---

## Generating Evaluation Figures

After training, run the standalone evaluator to generate publication-ready figures and a metrics table:

```bash
python scripts/evaluate_grn_model.py \
  --run-dir /path/to/outputs/gse318326_human_plasma \
  --results-dir /path/to/results/gse318326_eval \
  --top-features 4 \
  --top-edges 12
```

| Argument | Description |
|----------|-------------|
| `--run-dir` | Directory written by `run_pipeline.py` |
| `--results-dir` | Where to write figures and tables (created if absent) |
| `--top-features` | Genes per block to show in reconstruction plots (default: 4) |
| `--top-edges` | TF→target edges per trajectory in edge plots (default: 12) |

Generated outputs:

| File | Description |
|------|-------------|
| `figures/training_metrics.png` | Loss component breakdown over epochs |
| `figures/learned_delays.png` | Trajectories of τ₁ and τ₂ during training |
| `figures/{traj}_reconstructions.png` | Observed vs model z-scored trajectories per block |
| `figures/{traj}_edge_heatmap.png` | Heatmap of top TF→target weights across pseudotime |
| `figures/{traj}_edge_timecourses.png` | Time-course line plots for top edges |
| `tables/reconstruction_metrics.csv` | Per-trajectory per-block RMSE, MAE, R², Pearson r |
| `evaluation_summary.json` | Aggregate metrics and final training row |

---

## Running Everything at Once

`run_and_test_model.py` wraps training + evaluation + a smoke-test report in one command:

```bash
python run_and_test_model.py \
  --config configs/gse318326_run.yaml \
  --project gse318326_human_plasma \
  --run-dir outputs/gse318326_human_plasma \
  --results-dir results/gse318326_eval \
  --min-epochs 25
```

| Argument | Description |
|----------|-------------|
| `--config` | YAML config path |
| `--project` | Project name inside the config |
| `--run-dir` | Where training outputs go |
| `--results-dir` | Where evaluation figures/tables go |
| `--skip-train` | Reuse an existing run directory, skip training |
| `--min-epochs N` | Minimum epoch count for test to pass (default: 25) |
| `--max-loss FLOAT` | Optional ceiling on final training loss |
| `--extra-train-args ...` | Additional args forwarded to `run_pipeline.py` |

This writes a `test_report.json` to the results directory recording pass/fail for file-existence and training quality checks.

---

## Understanding the Outputs

After running `run_pipeline.py`, the output directory contains:

| File / Pattern | Description |
|----------------|-------------|
| `model.pt` | Trained PyTorch checkpoint |
| `training_history.csv` | Per-epoch: loss, regularization terms, learned delays |
| `condition_map.json` | Maps condition string → integer ID used internally |
| `selected_tfs.txt` | Ordered list of TF gene symbols used as features |
| `selected_targets.txt` | Ordered list of target gene symbols |
| `feature_stats.npz` | Normalization statistics for reproducible inference |
| `trajectory_manifest.csv` | One row per trajectory: condition, replicate, n_cells, n_bins |
| `top_dynamic_edges.csv` | Top TF→target edges at each pseudotime bin (long format) |
| `{traj}_pred_obs.npy` | Predicted states, shape `(n_bins, n_tfs*2 + n_targets)` |
| `{traj}_pred_edges.npy` | Full GRN tensor, shape `(n_bins, n_tfs, n_targets)` |
| `{traj}_pred_gates.npy` | TF activity gates over pseudotime, shape `(n_bins, n_tfs)` |
| `{traj}_tf_acc.npy` | Observed TF accessibility (binned), shape `(n_bins, n_tfs)` |
| `{traj}_tf_rna.npy` | Observed TF RNA (binned), shape `(n_bins, n_tfs)` |
| `{traj}_target_rna.npy` | Observed target RNA (binned), shape `(n_bins, n_targets)` |
| `{traj}_times.npy` | Normalized pseudotime values in [0, 1] |
| `{traj}_raw_times.npy` | Raw time labels (e.g. days) |

### Reading predictions in Python

```python
import numpy as np

run_dir = "outputs/gse318326_human_plasma"
tfs     = open(f"{run_dir}/selected_tfs.txt").read().splitlines()
targets = open(f"{run_dir}/selected_targets.txt").read().splitlines()
n_tfs   = len(tfs)

# Predicted trajectory  (n_bins × features)
pred_obs     = np.load(f"{run_dir}/DMSO_pred_obs.npy")
tf_acc_pred  = pred_obs[:, :n_tfs]           # TF accessibility
tf_rna_pred  = pred_obs[:, n_tfs:2*n_tfs]   # TF RNA
target_pred  = pred_obs[:, 2*n_tfs:]         # target RNA

# Dynamic GRN  (n_bins × n_tfs × n_targets)
edges = np.load(f"{run_dir}/DMSO_pred_edges.npy")
# edges[t, i, j]  =  weight of tfs[i] → targets[j] at pseudotime bin t
```

---

## In Silico TF Perturbation

Simulate knocking out a TF after training:

```bash
python run_pipeline.py \
  --config configs/gse318326_run.yaml \
  --project gse318326_human_plasma \
  --perturb-tf MEF2C \
  --perturb-start 0.45 \
  --perturb-mode both
```

This zeroes the TF's accessibility and/or RNA from `--perturb-start` onward and re-integrates the ODE. Results are written to `outputs/.../perturbations/`:

- `{traj}_{TF}_ko_pred_obs.npy` — full state trajectory under knockout
- `{traj}_{TF}_ko_pred_edges.npy` — GRN under knockout

Multiple TFs can be perturbed in one run:

```bash
python run_pipeline.py ... --perturb-tf MEF2C --perturb-tf IRF4
```

---

## Example Results (GSE318326)

Trained for 800 epochs on 8 time-point H5 files (days 4/7/10/13 × DMSO/A366), 100 TFs, 400 target genes, 32 pseudotime bins.

### Reconstruction metrics (median across trajectories)

| Block | RMSE | MAE | R² | Pearson r |
|-------|------|-----|----|-----------|
| TF accessibility | 0.0452 | 0.0304 | 0.920 | 0.961 |
| TF RNA | 0.0543 | 0.0360 | 0.995 | 0.998 |
| Target RNA | 0.0449 | 0.0307 | 0.994 | 0.997 |

### Learned regulatory delays (final epoch)

| Delay | Value (trajectory units) |
|-------|--------------------------|
| Accessibility → TF RNA (τ₁) | 0.035 |
| TF state → Target RNA (τ₂) | 0.050 |

These correspond to roughly 0.35–0.50 days of lag given the 13-day trajectory span.

### Output figures

Pre-generated figures are in `example_results/figures/`:

| Figure | Description |
|--------|-------------|
| `training_metrics.png` | Loss decomposition across 800 epochs |
| `learned_delays.png` | Convergence of τ₁ and τ₂ |
| `DMSO_reconstructions.png` | Model vs observed for DMSO trajectory |
| `A366_reconstructions.png` | Model vs observed for A366 trajectory |
| `DMSO_edge_heatmap.png` | Top TF→target dynamic weights — DMSO |
| `A366_edge_heatmap.png` | Top TF→target dynamic weights — A366 |
| `DMSO_edge_timecourses.png` | Time-course plots for top edges — DMSO |
| `A366_edge_timecourses.png` | Time-course plots for top edges — A366 |

---

## Repository Structure

```
GRN-Github/
├── README.md                         # This file
├── run_pipeline.py                   # Main training entrypoint
├── run_and_test_model.py             # Combined train + evaluate + test script
├── pyproject.toml                    # Package metadata
├── requirements.txt                  # Python dependencies
│
├── configs/
│   ├── example_project.yaml          # Template config (fill in your paths)
│   └── gse318326_run.yaml            # Config used to produce example results
│
├── scripts/
│   ├── evaluate_grn_model.py         # Standalone evaluation + figure generation
│   └── build_gse223041_metadata_template.py  # Helper for aggregated H5 datasets
│
├── src/
│   └── causal_grn/
│       ├── config.py                 # YAML config loading and dataclasses
│       ├── io_utils.py               # 10x H5 loading and metadata merging
│       ├── gtf_utils.py              # GTF parsing and promoter overlap mapping
│       ├── preprocess.py             # Filtering, normalization, gene activity
│       ├── pseudotime.py             # Joint embedding and graph pseudotime
│       ├── trajectory.py             # Feature selection and pseudotime binning
│       ├── model.py                  # Delay-aware Neural ODE GRN model (PyTorch)
│       └── train.py                  # Training loop, prediction, perturbation, export
│
└── example_results/
    ├── evaluation_summary.json       # Aggregate metrics from the GSE318326 run
    ├── reconstruction_metrics.csv    # Per-trajectory per-block metrics table
    └── figures/                      # Publication-ready evaluation figures
        ├── training_metrics.png
        ├── learned_delays.png
        ├── DMSO_reconstructions.png
        ├── A366_reconstructions.png
        ├── DMSO_edge_heatmap.png
        ├── A366_edge_heatmap.png
        ├── DMSO_edge_timecourses.png
        └── A366_edge_timecourses.png
```

## Practical Limitations

- Promoter accessibility is used as the TF accessibility proxy (not motif-derived activity)
- Cells are denoised by binning along pseudotime before fitting the ODE
- The solver is a fixed-step delay integrator, not an adaptive DDE solver
- Human and mouse projects are trained separately (different species/features)

## Suggested Next Improvements

- Replace promoter accessibility with motif-derived TF activity scores
- Add peak-to-gene priors from Cicero / ArchR / Signac / ABC
- Add a perturbation likelihood term if you have Perturb-seq labels
- Split trajectories by lineage branch instead of condition only
- Replace the fixed-step solver with a continuous-time latent variable model
