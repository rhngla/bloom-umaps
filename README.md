# bloom-umaps

Which UMAP settings best *illustrate* a Bloom embedding? A UMAP is always a distorted
picture; here it is used to show embeddings from models like Bloom, so we want maps where
cell classes are well separated but not collapsed to points.

**Scope:** one dataset and one embedding only: `allen_multiome` (`eval_balanced_50`,
15,750 cells, AIT33 subclass labels, 315 subclasses), Bloom RNA embedding
(`slurm_3458_step_530000`, `nz_top4096`, 1024-d, used as stored: no centring, scaling or PCA).

## Pipeline

    PY=/opt/dlami/nvme/envs/bloom-umaps/bin/python
    $PY scripts/01_sweep.py --pilot      # one fit under /usr/bin/time -v, to size the worker pool
    $PY scripts/01_sweep.py --jobs N     # all fits (cached; --force refits)
    $PY scripts/02_metrics.py --jobs N   # results/metrics.csv
    $PY scripts/03_plot.py               # results/nmi_vs_variance.png

| Step | Code | Output |
|---|---|---|
| UMAP sweep | `bloom_umaps.sweep` | `work/sweep/fits/<fit_id>.parquet` (nvme), `results/sweep_manifest.csv` |
| Metrics | `bloom_umaps.metrics` | `results/metrics.csv` |
| Plot | `bloom_umaps.plots` | `results/nmi_vs_variance.png` |

**Sweep grid** (ends inclusive): min_dist 0.2:0.2:1.6, n_neighbors 5:5:25, spread 1.0:0.5:2.0,
keeping only min_dist <= spread (umap-learn rejects the rest): 100 fits. Every fit is
`umap.UMAP(n_neighbors, min_dist, spread, metric="euclidean", random_state=0)`, all other
arguments at umap-learn 0.5.12 defaults.

**Parallelism:** `random_state` makes umap-learn run each fit single-threaded (it overrides
`n_jobs`), so fits run one per process with joblib's default `loky` backend, which also caps
each worker's BLAS / OpenMP / numba threads. Size `--jobs` from the pilot's peak RSS.

## Metrics (one row per fit)

* **Seeded k-means NMI** -- `sklearn.cluster.KMeans(n_clusters=K, init=C, n_init=1)` on the 2-D
  coordinates, K = number of subclasses, C = one randomly drawn cell per subclass;
  `sklearn.metrics.normalized_mutual_info_score` against AIT33 subclass. Mean and sd over
  seeds 0, 1, 2 (the seed only picks the initial cells). The label-seeded initialisation makes
  this higher than unsupervised k-means; use it to RANK maps, not to compare with other NMIs.
* **Within-subclass variance**, two normalisations (both isotropic, neither whitens):
  * `v_span` = mean over subclasses of trace(cov_c) / span^2, where span is exactly how
    bloom-figures' `umap_frame` scales a panel (1.06 x the larger of the x / y 0.2-99.8
    percentile ranges). This is the spread a viewer sees, since every panel is autoscaled.
  * `v_total` = mean over subclasses of trace(cov_c) / trace(cov_all).

## Storage

All stored embeddings / coordinates live on nvme only: `work/` is a gitignored symlink to
`/opt/dlami/nvme/work/bloom-umaps` (instance store: survives reboot, wiped on stop; covered
by the nightly S3 sync). Git holds code, the manifest and the metrics table.
