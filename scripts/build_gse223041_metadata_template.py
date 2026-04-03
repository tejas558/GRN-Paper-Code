#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from causal_grn.io_utils import read_10x_h5_multimodal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a barcode-level metadata template for an aggregated 10x multiome H5 file."
    )
    parser.add_argument("--h5", required=True, help="Path to GSE223041_Aggregated_filtered_feature_bc_matrix.h5")
    parser.add_argument("--out", required=True, help="Where to write the CSV template")
    return parser.parse_args()


def infer_library_id(barcode: str) -> str:
    parts = barcode.split("-")
    if len(parts) >= 3:
        return parts[-1]
    if len(parts) == 2:
        return parts[-1]
    return "unknown"


def main() -> None:
    args = parse_args()
    tx = read_10x_h5_multimodal(args.h5)
    barcodes = [str(x) for x in tx.barcodes]
    df = pd.DataFrame(
        {
            "barcode": barcodes,
            "library_id": [infer_library_id(x) for x in barcodes],
            "time": "",
            "condition": "AA_LiCl",
            "replicate": "",
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote metadata template to {out}")


if __name__ == "__main__":
    main()
