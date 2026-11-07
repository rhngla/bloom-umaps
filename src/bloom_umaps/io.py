"""Inputs: the Bloom RNA embedding and AIT33 subclass labels of allen_multiome."""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np

EMBEDDING = Path(
    "/mnt/bloom-output/experiments/results/slurm_3458_step_530000/"
    "eval_balanced_50_allen_multiome_nz_top4096/embeddings.h5ad"
)
METADATA = Path(
    "/mnt/bloom-output/datasets/hyacinth-v1/eval_balanced_50/allen_multiome/metadata.h5ad"
)
LABEL_COL = "AIT33_subclass"
N_CELLS = 15_750


def read_embedding(path: Path = EMBEDDING) -> np.ndarray:
    """Bloom embedding as stored (obsm/bloom_emb), float32, positional row order."""
    with h5py.File(path, "r") as f:
        X = f["obsm"]["bloom_emb"][:].astype("float32")
    if X.shape[0] != N_CELLS:
        raise SystemExit(f"expected {N_CELLS} rows, got {X.shape}")
    return X


def read_labels(path: Path = METADATA, col: str = LABEL_COL) -> np.ndarray:
    """Subclass label per cell, aligned to the embedding BY POSITION (no ids in Bloom files)."""
    import anndata as ad

    obs = ad.read_h5ad(path, backed="r").obs
    lab = obs[col]
    if lab.isna().any():
        raise SystemExit(f"{int(lab.isna().sum())} cells have no {col}")
    lab = lab.astype(str).to_numpy()
    if len(lab) != N_CELLS:
        raise SystemExit(f"expected {N_CELLS} labels, got {len(lab)}")
    return lab


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    """Hex sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()
