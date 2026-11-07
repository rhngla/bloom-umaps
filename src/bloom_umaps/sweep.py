"""UMAP parameter sweep: one euclidean fit per valid (n_neighbors, min_dist, spread)."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from .paths import FITS

N_NEIGHBORS = [5, 10, 15, 20, 25]  # 5:5:25
MIN_DIST = [round(0.2 * i, 1) for i in range(1, 9)]  # 0.2:0.2:1.6
SPREAD = [1.0, 1.5, 2.0]  # 1.0:0.5:2.0
METRIC, SEED = "euclidean", 0


def fit_id(n_neighbors: int, min_dist: float, spread: float) -> str:
    """Stable file stem, e.g. nn05_md0.2_sp1.0."""
    return f"nn{n_neighbors:02d}_md{min_dist:.1f}_sp{spread:.1f}"


def grid() -> pd.DataFrame:
    """All valid parameter combinations (umap-learn requires min_dist <= spread)."""
    rows = [
        dict(n_neighbors=nn, min_dist=md, spread=sp, fit_id=fit_id(nn, md, sp))
        for nn, md, sp in itertools.product(N_NEIGHBORS, MIN_DIST, SPREAD)
        if md <= sp + 1e-9
    ]
    return pd.DataFrame(rows)


def fit_one(
    X: np.ndarray, n_neighbors: int, min_dist: float, spread: float, force: bool = False
) -> dict:
    """Fit one UMAP and cache its 2-D coordinates as work/sweep/fits/<fit_id>.parquet."""
    import umap

    fid = fit_id(n_neighbors, min_dist, spread)
    out = FITS / f"{fid}.parquet"
    if out.exists() and not force:
        return dict(fit_id=fid, path=str(out), seconds=np.nan, cached=True)
    t0 = time.time()
    Y = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric=METRIC,
        random_state=SEED,
    ).fit_transform(X)
    dt = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    pd.DataFrame(
        {
            "cell_row": np.arange(len(Y)),
            "umap_1": Y[:, 0].astype("float32"),
            "umap_2": Y[:, 1].astype("float32"),
        }
    ).to_parquet(tmp, index=False)
    tmp.replace(out)
    return dict(fit_id=fid, path=str(out), seconds=round(dt, 1), cached=False)


def read_fit(fid: str) -> np.ndarray:
    """Cached coordinates of one fit, (n_cells, 2), in cell_row order."""
    df = pd.read_parquet(FITS / f"{fid}.parquet").sort_values("cell_row")
    return df[["umap_1", "umap_2"]].to_numpy()
