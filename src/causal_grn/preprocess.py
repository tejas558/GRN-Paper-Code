from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

from .config import PreprocessConfig
from .gtf_utils import build_peak_to_gene_map
from .io_utils import LoadedSample


@dataclass
class ProcessedSample:
    sample_name: str
    condition: str
    replicate: str
    time: float
    obs: pd.DataFrame
    rna: sp.csr_matrix
    rna_genes: np.ndarray
    gene_activity: sp.csr_matrix
    ga_genes: np.ndarray


@dataclass
class ProjectCellData:
    project_name: str
    species: str
    obs: pd.DataFrame
    rna: np.ndarray
    ga: np.ndarray
    genes: np.ndarray


@dataclass
class SelectedFeatures:
    tf_names: List[str]
    target_names: List[str]
    tf_indices: np.ndarray
    target_indices: np.ndarray


EPS = 1e-8


def _row_sums(X: sp.spmatrix) -> np.ndarray:
    return np.asarray(X.sum(axis=1)).ravel()


def _col_nnz(X: sp.spmatrix) -> np.ndarray:
    return np.asarray((X > 0).sum(axis=0)).ravel()


def filter_cells_and_features(
    rna: sp.csr_matrix,
    atac: sp.csr_matrix,
    rna_genes: np.ndarray,
    peak_names: np.ndarray,
    cfg: PreprocessConfig,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    rna = rna.tocsr().astype(np.float32)
    atac = atac.tocsr().astype(np.float32)

    n_rna = _row_sums(rna)
    n_atac = _row_sums(atac)
    detected_genes = np.asarray((rna > 0).sum(axis=1)).ravel()
    detected_peaks = np.asarray((atac > 0).sum(axis=1)).ravel()

    keep_cells = (
        (n_rna >= cfg.min_rna_counts)
        & (n_atac >= cfg.min_atac_counts)
        & (detected_genes >= cfg.min_detected_genes)
        & (detected_peaks >= cfg.min_detected_peaks)
    )
    rna = rna[keep_cells].tocsr()
    atac = atac[keep_cells].tocsr()

    keep_genes = _col_nnz(rna) >= cfg.min_cells_per_gene
    keep_peaks = _col_nnz(atac) >= cfg.min_cells_per_peak
    rna = rna[:, keep_genes].tocsr()
    atac = atac[:, keep_peaks].tocsr()
    return rna, atac, rna_genes[keep_genes], peak_names[keep_peaks], keep_cells


def normalize_log1p_sparse(X: sp.csr_matrix, target_sum: float = 1e4) -> sp.csr_matrix:
    X = X.tocsr().astype(np.float32).copy()
    sums = _row_sums(X)
    sums[sums == 0.0] = 1.0
    scale = target_sum / sums
    X = sp.diags(scale.astype(np.float32)) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def sparse_to_dense(X: sp.csr_matrix) -> np.ndarray:
    return np.asarray(X.toarray(), dtype=np.float32)


def compute_gene_activity(
    atac_counts: sp.csr_matrix,
    peak_names: np.ndarray,
    gtf_path: str,
    promoter_upstream: int,
    promoter_downstream: int,
) -> tuple[sp.csr_matrix, np.ndarray]:
    mapping, gene_names = build_peak_to_gene_map(
        peak_names=peak_names,
        gtf_path=gtf_path,
        upstream=promoter_upstream,
        downstream=promoter_downstream,
    )
    gene_activity = atac_counts @ mapping
    gene_activity = gene_activity.tocsr().astype(np.float32)
    # Collapse duplicated gene symbols by summing columns via a sparse merge matrix.
    unique_genes, gene_to_uniq = np.unique(gene_names, return_inverse=True)
    n_promoters = len(gene_names)
    n_unique = len(unique_genes)
    merge_mat = sp.csr_matrix(
        (np.ones(n_promoters, dtype=np.float32), (np.arange(n_promoters), gene_to_uniq)),
        shape=(n_promoters, n_unique),
    )
    collapsed = (gene_activity @ merge_mat).tocsr().astype(np.float32)
    return collapsed, unique_genes.astype(object)


def preprocess_sample(sample: LoadedSample, gtf_path: str, cfg: PreprocessConfig) -> ProcessedSample:
    rna, atac, genes, peaks, keep_cells = filter_cells_and_features(
        sample.rna_counts,
        sample.atac_counts,
        sample.rna_genes,
        sample.peak_names,
        cfg,
    )
    obs = sample.obs.iloc[np.where(keep_cells)[0]].copy()
    obs.index = obs["barcode"].astype(str)

    rna_norm = normalize_log1p_sparse(rna)
    ga, ga_genes = compute_gene_activity(
        atac_counts=atac,
        peak_names=peaks,
        gtf_path=gtf_path,
        promoter_upstream=cfg.promoter_upstream,
        promoter_downstream=cfg.promoter_downstream,
    )
    ga_norm = normalize_log1p_sparse(ga)

    if obs["time"].isna().any():
        raise ValueError(
            f"Sample {sample.sample_name} contains cells without time labels after metadata merge."
        )

    time_val = float(pd.to_numeric(obs["time"], errors="coerce").dropna().iloc[0])
    return ProcessedSample(
        sample_name=sample.sample_name,
        condition=str(obs["condition"].iloc[0]),
        replicate=str(obs["replicate"].iloc[0]),
        time=time_val,
        obs=obs,
        rna=rna_norm,
        rna_genes=genes,
        gene_activity=ga_norm,
        ga_genes=ga_genes,
    )


def _intersect_ordered(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    bset = set(str(x) for x in b)
    return np.asarray([str(x) for x in a if str(x) in bset], dtype=object)


def _reindex_columns(
    X: sp.csr_matrix,
    old_names: np.ndarray,
    new_names: np.ndarray,
) -> sp.csr_matrix:
    index = {str(name): i for i, name in enumerate(old_names.astype(str))}
    cols = [index[str(name)] for name in new_names]
    return X[:, cols].tocsr()


def merge_processed_samples(
    samples: List[ProcessedSample],
    project_name: str,
    species: str,
) -> ProjectCellData:
    if not samples:
        raise ValueError("No processed samples to merge.")
    common_genes = samples[0].rna_genes.astype(str)
    common_ga_genes = samples[0].ga_genes.astype(str)
    for sample in samples[1:]:
        common_genes = _intersect_ordered(common_genes, sample.rna_genes.astype(str))
        common_ga_genes = _intersect_ordered(common_ga_genes, sample.ga_genes.astype(str))
    common = _intersect_ordered(common_genes, common_ga_genes)
    if len(common) < 100:
        raise ValueError(
            "Fewer than 100 genes overlap between RNA and promoter accessibility across samples. "
            "Check that all samples belong to the same species and use the same genome build."
        )

    obs_tables: List[pd.DataFrame] = []
    rna_blocks: List[np.ndarray] = []
    ga_blocks: List[np.ndarray] = []
    for sample in samples:
        rna_sub = _reindex_columns(sample.rna, sample.rna_genes, common)
        ga_sub = _reindex_columns(sample.gene_activity, sample.ga_genes, common)
        obs = sample.obs.copy()
        obs["sample_time"] = sample.time
        obs["condition"] = sample.condition
        obs["replicate"] = sample.replicate
        obs["sample_name"] = sample.sample_name
        obs_tables.append(obs)
        rna_blocks.append(sparse_to_dense(rna_sub))
        ga_blocks.append(sparse_to_dense(ga_sub))

    obs_all = pd.concat(obs_tables, axis=0)
    obs_all.index = obs_all["barcode"].astype(str)
    rna_all = np.concatenate(rna_blocks, axis=0).astype(np.float32)
    ga_all = np.concatenate(ga_blocks, axis=0).astype(np.float32)

    return ProjectCellData(
        project_name=project_name,
        species=species,
        obs=obs_all,
        rna=rna_all,
        ga=ga_all,
        genes=common.astype(object),
    )


def _variance_score(X: np.ndarray) -> np.ndarray:
    return X.var(axis=0).astype(np.float32)


def select_features(
    data: ProjectCellData,
    cfg: PreprocessConfig,
    tf_list: Optional[Iterable[str]] = None,
) -> SelectedFeatures:
    genes = np.asarray(data.genes.astype(str), dtype=object)
    variances = _variance_score(data.rna)
    ranking = np.argsort(-variances)
    ranked_genes = genes[ranking]

    if tf_list is not None:
        tf_set = {str(x) for x in tf_list}
        tf_candidates = [g for g in ranked_genes if g in tf_set]
    else:
        tf_candidates = [g for g in ranked_genes]

    tf_names = tf_candidates[: cfg.n_tfs]
    if len(tf_names) < min(10, cfg.n_tfs):
        raise ValueError(
            "Too few TF candidates were found. Supply a TF list matching the species if you want biologically meaningful regulators."
        )
    tf_set = set(tf_names)
    target_names = [g for g in ranked_genes if g not in tf_set][: cfg.n_targets]

    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    tf_idx = np.asarray([gene_to_idx[g] for g in tf_names], dtype=int)
    target_idx = np.asarray([gene_to_idx[g] for g in target_names], dtype=int)

    return SelectedFeatures(
        tf_names=list(tf_names),
        target_names=list(target_names),
        tf_indices=tf_idx,
        target_indices=target_idx,
    )


def zscore_features(X: np.ndarray) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xz = scaler.fit_transform(X).astype(np.float32)
    stats = {
        "mean": scaler.mean_.astype(np.float32),
        "scale": np.where(scaler.scale_ < EPS, 1.0, scaler.scale_).astype(np.float32),
    }
    return Xz, stats
