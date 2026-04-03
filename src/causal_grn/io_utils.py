from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import ProjectSpec, SampleSpec


@dataclass
class LoadedSample:
    project_name: str
    sample_name: str
    species: str
    condition: str
    replicate: str
    time: Optional[float]
    obs: pd.DataFrame
    rna_counts: sp.csr_matrix
    rna_genes: np.ndarray
    atac_counts: sp.csr_matrix
    peak_names: np.ndarray


@dataclass
class CellLevelProject:
    project_name: str
    species: str
    obs: pd.DataFrame
    rna: sp.csr_matrix
    rna_genes: np.ndarray
    gene_activity: sp.csr_matrix
    gene_activity_genes: np.ndarray


@dataclass
class ProjectTrajectories:
    project_name: str
    species: str
    conditions: List[str]
    tf_names: List[str]
    target_names: List[str]
    trajectories: List[dict]
    feature_stats: Dict[str, np.ndarray]


@dataclass
class TenXMatrix:
    barcodes: np.ndarray
    feature_names: np.ndarray
    feature_types: np.ndarray
    matrix: sp.csc_matrix
    feature_ids: np.ndarray


_DEF_GENE_KEYS = ["Gene Expression", "gene expression"]
_DEF_ATAC_KEYS = ["Peaks", "peaks", "ATAC", "Accessibility"]


def _decode_array(values: Sequence) -> np.ndarray:
    out: List[str] = []
    for value in values:
        if isinstance(value, (bytes, np.bytes_)):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return np.asarray(out, dtype=object)


def make_unique(names: Sequence[str]) -> np.ndarray:
    counts: Dict[str, int] = {}
    unique: List[str] = []
    for name in names:
        if name not in counts:
            counts[name] = 0
            unique.append(name)
        else:
            counts[name] += 1
            unique.append(f"{name}_{counts[name]}")
    return np.asarray(unique, dtype=object)


def read_10x_h5_multimodal(path: str | Path) -> TenXMatrix:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if "matrix" not in handle:
            raise ValueError(f"{path} does not look like a 10x filtered_feature_bc_matrix.h5 file")
        grp = handle["matrix"]
        barcodes = _decode_array(grp["barcodes"][:])
        data = grp["data"][:]
        indices = grp["indices"][:]
        indptr = grp["indptr"][:]
        shape = tuple(int(x) for x in grp["shape"][:])
        matrix = sp.csc_matrix((data, indices, indptr), shape=shape)

        feat_grp = grp["features"]
        feature_names = _decode_array(feat_grp["name"][:])
        feature_ids = _decode_array(feat_grp["id"][:])
        feature_types = _decode_array(feat_grp["feature_type"][:])

    return TenXMatrix(
        barcodes=barcodes,
        feature_names=make_unique(feature_names),
        feature_types=feature_types,
        matrix=matrix,
        feature_ids=feature_ids,
    )


def _find_feature_type_mask(feature_types: np.ndarray, candidates: Sequence[str]) -> np.ndarray:
    lowered = np.char.lower(feature_types.astype(str))
    for candidate in candidates:
        mask = lowered == candidate.lower()
        if mask.any():
            return mask
    return np.zeros_like(lowered, dtype=bool)


def split_modalities(tx: TenXMatrix) -> tuple[sp.csr_matrix, np.ndarray, sp.csr_matrix, np.ndarray]:
    gene_mask = _find_feature_type_mask(tx.feature_types, _DEF_GENE_KEYS)
    atac_mask = _find_feature_type_mask(tx.feature_types, _DEF_ATAC_KEYS)
    if not gene_mask.any():
        raise ValueError("Could not find Gene Expression features in the 10x H5 file.")
    if not atac_mask.any():
        raise ValueError("Could not find ATAC peak features in the 10x H5 file.")

    rna = tx.matrix[gene_mask, :].T.tocsr().astype(np.float32)
    atac = tx.matrix[atac_mask, :].T.tocsr().astype(np.float32)
    genes = tx.feature_names[gene_mask]
    peaks = tx.feature_names[atac_mask]
    return rna, genes, atac, peaks


def _default_sample_name(spec: SampleSpec) -> str:
    if spec.sample_name:
        return spec.sample_name
    return Path(spec.path).stem.replace("_filtered_feature_bc_matrix", "")


def _read_metadata_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"barcode"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata file {path} is missing required columns: {sorted(missing)}")
    return df


def load_sample(spec: SampleSpec, project_name: str, species: str) -> LoadedSample:
    tx = read_10x_h5_multimodal(spec.path)
    rna, genes, atac, peaks = split_modalities(tx)
    sample_name = _default_sample_name(spec)

    obs = pd.DataFrame(index=pd.Index(tx.barcodes, name="barcode"))
    obs["barcode"] = tx.barcodes
    obs["project"] = project_name
    obs["sample_name"] = sample_name
    obs["species"] = species
    obs["condition"] = spec.condition
    obs["replicate"] = spec.replicate
    obs["time"] = spec.time
    obs["source_h5"] = str(spec.path)

    if spec.kind.lower() == "aggregated":
        if not spec.metadata_csv:
            raise ValueError(
                f"Sample {sample_name} is marked as aggregated, but metadata_csv was not provided. "
                "For aggregated 10x aggr H5 files you need a barcode-level metadata CSV with at least a 'barcode' column "
                "and preferably 'time', 'condition', and 'replicate'."
            )
        md = _read_metadata_csv(spec.metadata_csv).copy()
        md["barcode"] = md["barcode"].astype(str)
        obs = obs.merge(md, on="barcode", how="left", suffixes=("", "_meta"))
        obs.index = obs["barcode"].astype(str)
        # Prioritise metadata values when present.
        for column in ["time", "condition", "replicate"]:
            meta_col = f"{column}_meta"
            if meta_col in obs.columns:
                obs[column] = obs[meta_col].where(obs[meta_col].notna(), obs[column])
                obs = obs.drop(columns=[meta_col])
        if obs["time"].isna().all():
            raise ValueError(
                f"No per-cell time labels were recovered for aggregated sample {sample_name}. "
                "Fill in the metadata CSV with at least 'barcode' and 'time' columns."
            )
    else:
        obs.index = obs["barcode"].astype(str)

    return LoadedSample(
        project_name=project_name,
        sample_name=sample_name,
        species=species,
        condition=str(obs["condition"].iloc[0]),
        replicate=str(obs["replicate"].iloc[0]),
        time=float(obs["time"].iloc[0]) if pd.notnull(obs["time"].iloc[0]) else None,
        obs=obs,
        rna_counts=rna,
        rna_genes=genes,
        atac_counts=atac,
        peak_names=peaks,
    )


def load_project_samples(project: ProjectSpec) -> List[LoadedSample]:
    if not project.samples:
        raise ValueError(f"Project {project.name} has no samples in the config file.")
    loaded: List[LoadedSample] = []
    for spec in project.samples:
        loaded.append(load_sample(spec, project.name, project.species))
    return loaded


def read_tf_list(path: Optional[str | Path]) -> Optional[List[str]]:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    genes: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            gene = line.strip()
            if gene:
                genes.append(gene)
    return genes
