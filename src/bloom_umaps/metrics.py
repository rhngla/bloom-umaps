"""Map-quality metrics on 2-D coordinates: seeded k-means NMI and within-class variance."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

SEEDS = (0, 1, 2)
FRAME_PAD, FRAME_Q = 1.06, (0.2, 99.8)  # bloom-figures plotstyle.umap_frame defaults


def seeded_init(Y: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """One randomly drawn cell per class (classes in sorted order) as k-means centroids."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    return np.stack([Y[rng.choice(np.flatnonzero(labels == c))] for c in classes])


def seeded_kmeans_nmi(Y: np.ndarray, labels: np.ndarray, seed: int) -> float:
    """NMI between subclass labels and k-means (K = n classes) seeded with one cell per class."""
    init = seeded_init(Y, labels, seed)
    km = KMeans(n_clusters=len(init), init=init, n_init=1, random_state=seed).fit(Y)
    return float(normalized_mutual_info_score(labels, km.labels_))


def frame_span(Y: np.ndarray, pad: float = FRAME_PAD, q=FRAME_Q) -> float:
    """Displayed span of a panel, exactly as bloom-figures umap_frame computes it."""
    lo, hi = np.percentile(Y, q, axis=0)
    return float(pad * np.max(hi - lo))


def within_class_trace(Y: np.ndarray, labels: np.ndarray) -> float:
    """Mean over classes of trace(cov_c) (population variance, summed over the 2 axes)."""
    return float(np.mean([Y[labels == c].var(axis=0).sum() for c in np.unique(labels)]))


def variance_metrics(Y: np.ndarray, labels: np.ndarray) -> dict:
    """v_span = W / span^2 (what a viewer sees); v_total = W / trace(cov_all)."""
    w = within_class_trace(Y, labels)
    return dict(
        v_span=w / frame_span(Y) ** 2,
        v_total=w / float(Y.var(axis=0).sum()),
        within_trace_raw=w,
        span=frame_span(Y),
    )


def all_metrics(Y: np.ndarray, labels: np.ndarray, seeds=SEEDS) -> dict:
    """Every metric for one map; NMI reported per seed, as mean and as sd (ddof=1)."""
    nmi = [seeded_kmeans_nmi(Y, labels, s) for s in seeds]
    out = {f"nmi_seed{s}": v for s, v in zip(seeds, nmi, strict=True)}
    out.update(nmi_mean=float(np.mean(nmi)), nmi_sd=float(np.std(nmi, ddof=1)))
    out.update(variance_metrics(Y, labels))
    return out
