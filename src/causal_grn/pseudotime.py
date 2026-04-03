from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def build_joint_embedding(
    rna: np.ndarray,
    ga: np.ndarray,
    n_rna_components: int = 30,
    n_ga_components: int = 30,
) -> np.ndarray:
    n_rna_components = min(n_rna_components, rna.shape[0] - 1, rna.shape[1] - 1)
    n_ga_components = min(n_ga_components, ga.shape[0] - 1, ga.shape[1] - 1)
    n_rna_components = max(2, n_rna_components)
    n_ga_components = max(2, n_ga_components)

    rna_emb = PCA(n_components=n_rna_components, random_state=0).fit_transform(rna)
    ga_emb = PCA(n_components=n_ga_components, random_state=0).fit_transform(ga)
    emb = np.concatenate([rna_emb, ga_emb], axis=1)
    emb = StandardScaler(with_mean=True, with_std=True).fit_transform(emb)
    return emb.astype(np.float32)


def _make_knn_graph(embedding: np.ndarray, n_neighbors: int = 20) -> sp.csr_matrix:
    n_neighbors = min(n_neighbors, max(2, embedding.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(embedding)
    distances, indices = nn.kneighbors(embedding)

    rows = np.repeat(np.arange(embedding.shape[0]), n_neighbors)
    cols = indices.reshape(-1)
    vals = distances.reshape(-1)
    graph = sp.csr_matrix((vals, (rows, cols)), shape=(embedding.shape[0], embedding.shape[0]))
    graph = graph.minimum(graph.T)
    return graph


def compute_graph_pseudotime(
    embedding: np.ndarray,
    actual_time: np.ndarray,
    n_neighbors: int = 20,
) -> np.ndarray:
    graph = _make_knn_graph(embedding, n_neighbors=n_neighbors)
    earliest_time = np.min(actual_time)
    root_mask = actual_time == earliest_time
    if root_mask.sum() == 0:
        raise ValueError("Could not identify a root cell from the earliest time point.")
    root_embedding = embedding[root_mask]
    centroid = root_embedding.mean(axis=0)
    root_idx_candidates = np.where(root_mask)[0]
    local_root = np.argmin(np.sum((root_embedding - centroid[None, :]) ** 2, axis=1))
    root_idx = int(root_idx_candidates[local_root])

    dist = dijkstra(graph, directed=False, indices=root_idx)
    finite = np.isfinite(dist)
    if finite.sum() < len(dist):
        max_finite = np.max(dist[finite]) if finite.any() else 1.0
        dist[~finite] = max_finite
    dist = dist - dist.min()
    dist = dist / max(dist.max(), 1e-8)
    return dist.astype(np.float32)


def continuous_time_from_discrete_and_pseudotime(
    actual_time: np.ndarray,
    pseudotime: np.ndarray,
) -> np.ndarray:
    actual_time = np.asarray(actual_time, dtype=float)
    pseudotime = np.asarray(pseudotime, dtype=float)
    uniq = np.sort(np.unique(actual_time))
    if len(uniq) == 1:
        return pseudotime.copy()

    gaps = np.diff(uniq)
    median_gap = float(np.median(gaps))
    out = np.zeros_like(actual_time, dtype=float)
    for i, t in enumerate(uniq):
        mask = actual_time == t
        local = pseudotime[mask]
        if local.size == 1:
            q = np.array([0.5], dtype=float)
        else:
            q = (rankdata(local, method="average") - 1.0) / (local.size - 1.0)
        gap = gaps[i] if i < len(gaps) else median_gap
        out[mask] = t + 0.9 * gap * q
    order = np.argsort(out)
    out[order] = np.maximum.accumulate(out[order])
    return out.astype(np.float32)


def normalise_time(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = values.min()
    vmax = values.max()
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)
