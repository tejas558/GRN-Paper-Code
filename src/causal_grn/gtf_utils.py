from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pyranges as pr
import scipy.sparse as sp


GTF_COLUMNS = [
    "Chromosome",
    "Source",
    "Feature",
    "Start",
    "End",
    "Score",
    "Strand",
    "Frame",
    "Attributes",
]


def _normalise_chr_names(values: pd.Series) -> pd.Series:
    vals = values.astype(str)
    has_chr = vals.str.startswith("chr").mean() > 0.5
    fixed = vals.copy()
    if has_chr:
        fixed = fixed.where(fixed.str.startswith("chr"), "chr" + fixed)
    else:
        fixed = fixed.str.replace("^chr", "", regex=True)
    return fixed


def read_gene_table(gtf_path: str | Path) -> pd.DataFrame:
    path = Path(gtf_path)
    if not path.exists():
        raise FileNotFoundError(path)
    gtf = pr.read_gtf(str(path))
    df = gtf.as_df()
    if "Feature" not in df.columns:
        raise ValueError(f"GTF file {gtf_path} does not contain a Feature column.")
    genes = df.loc[df["Feature"] == "gene"].copy()
    if genes.empty:
        raise ValueError("No gene features found in the supplied GTF file.")
    if "gene_name" not in genes.columns:
        raise ValueError("The GTF file does not contain gene_name annotations.")
    genes["Chromosome"] = _normalise_chr_names(genes["Chromosome"])
    genes["gene_name"] = genes["gene_name"].astype(str)
    genes = genes.drop_duplicates(subset=["gene_name", "Chromosome", "Start", "End"])
    return genes[["Chromosome", "Start", "End", "Strand", "gene_id", "gene_name"]].reset_index(drop=True)


def peak_names_to_df(peak_names: np.ndarray) -> pd.DataFrame:
    chroms = []
    starts = []
    ends = []
    for idx, peak in enumerate(peak_names.astype(str)):
        name = peak.replace("-", ":", 1) if peak.count(":") == 0 and peak.count("-") >= 2 else peak
        if ":" not in name or "-" not in name:
            raise ValueError(
                "Peak names must look like 'chr1:1000-1500'. "
                f"Offending peak: {peak}"
            )
        chrom, rest = name.split(":", 1)
        start_str, end_str = rest.split("-", 1)
        chroms.append(chrom)
        starts.append(int(start_str))
        ends.append(int(end_str))
    df = pd.DataFrame(
        {
            "Chromosome": _normalise_chr_names(pd.Series(chroms)),
            "Start": starts,
            "End": ends,
            "peak_idx": np.arange(len(peak_names), dtype=int),
            "peak_name": peak_names.astype(str),
        }
    )
    return df


def promoter_windows(genes: pd.DataFrame, upstream: int = 2000, downstream: int = 500) -> pd.DataFrame:
    promoters = genes.copy()
    pos = promoters["Strand"] == "+"
    neg = ~pos

    promoters.loc[pos, "promoter_start"] = (promoters.loc[pos, "Start"] - upstream).clip(lower=0)
    promoters.loc[pos, "promoter_end"] = promoters.loc[pos, "Start"] + downstream

    promoters.loc[neg, "promoter_start"] = (promoters.loc[neg, "End"] - downstream).clip(lower=0)
    promoters.loc[neg, "promoter_end"] = promoters.loc[neg, "End"] + upstream

    promoters["promoter_start"] = promoters["promoter_start"].astype(int)
    promoters["promoter_end"] = promoters["promoter_end"].astype(int)
    return promoters


def build_peak_to_gene_map(
    peak_names: np.ndarray,
    gtf_path: str | Path,
    upstream: int = 2000,
    downstream: int = 500,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    genes = read_gene_table(gtf_path)
    peaks_df = peak_names_to_df(peak_names)

    # Reconcile chromosome naming: force GTF to match the convention used by peaks
    use_chr = peaks_df["Chromosome"].str.startswith("chr").mean() > 0.5
    if use_chr:
        genes["Chromosome"] = genes["Chromosome"].where(
            genes["Chromosome"].str.startswith("chr"),
            "chr" + genes["Chromosome"],
        )
    else:
        genes["Chromosome"] = genes["Chromosome"].str.replace("^chr", "", regex=True)

    promoters = promoter_windows(genes, upstream=upstream, downstream=downstream).copy()
    promoters["gene_idx"] = np.arange(len(promoters), dtype=int)
    peaks_gr = pr.PyRanges(peaks_df[["Chromosome", "Start", "End", "peak_idx", "peak_name"]])
    prom_gr = pr.PyRanges(
        promoters[
            [
                "Chromosome",
                "promoter_start",
                "promoter_end",
                "gene_idx",
                "gene_name",
            ]
        ].rename(columns={"promoter_start": "Start", "promoter_end": "End"})
    )

    joined = peaks_gr.join(prom_gr).as_df()
    if joined.empty:
        raise ValueError(
            "No ATAC peaks overlapped promoter windows. Check that the GTF genome build matches the ATAC peak coordinates."
        )

    rows = joined["peak_idx"].to_numpy(dtype=int)
    cols = joined["gene_idx"].to_numpy(dtype=int)
    data = np.ones(len(joined), dtype=np.float32)
    mapping = sp.csr_matrix((data, (rows, cols)), shape=(len(peak_names), len(promoters)))
    # Average over the number of peaks assigned to each gene so the scale is roughly comparable.
    gene_peak_counts = np.asarray(mapping.sum(axis=0)).ravel()
    gene_peak_counts[gene_peak_counts == 0.0] = 1.0
    mapping = mapping @ sp.diags(1.0 / gene_peak_counts.astype(np.float32))

    gene_names = promoters["gene_name"].astype(str).to_numpy()
    return mapping, gene_names
