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

## Dashboard

Interactive explorer: **https://rhngla.github.io/bloom-umaps/** (GitHub Pages, `main:/docs`).
Left, every fit's NMI against within-subclass variance (Observable Plot): hover previews a
fit, click pins it. Right, the selected UMAP of all 15,750 cells (Apple
[Embedding Atlas](https://apple.github.io/embedding-atlas/) `EmbeddingView`), coloured by AIT33
subclass or class, with a per-cell tooltip. Zoom and pan are kept across fits, relative to
each fit's default frame (fits do not share a coordinate system); "reset view" returns to the
whole map.

* Needs a browser with WebGPU and the `shader-f16` feature (recent desktop Chrome / Edge,
  Safari 26+). Embedding Atlas has no WebGL fallback; other browsers get a message in the
  map panel and the metrics panel still works.
* Labels are the metadata's own `AIT33_subclass` / `AIT33_class` (128 cells carry a class that
  is not their subclass's parent; the tooltip flags them).
* EmbeddingView colours by a Uint8 category (max 256), so the 59 subclass pairs with the
  closest taxonomy colours (CIE76 dE, cross-class pairs only) share a colour; the list is in
  `results/dashboard_subclass_colour_merges.csv`. Tooltips always name the exact subclass.
* Only 2-D UMAP coordinates (uint16, error <= 1/131070 of a map's extent) and labels are
  published; the Bloom embedding itself is not.

Rebuild after changing the sweep or metrics:

    python scripts/04_export_dashboard.py          # docs/data
    cd web && npm install && npm run build         # docs/index.html + docs/assets (Node >= 20)

## Storage

The full-precision sweep fits live on nvme only (the one exception is the dashboard export:
`docs/data` holds uint16-quantised 2-D coordinates and labels, committed so Pages can serve
them). `work/` is a gitignored symlink to
`/opt/dlami/nvme/work/bloom-umaps` (instance store: survives reboot, wiped on stop; covered
by the nightly S3 sync). Git holds code, the manifest and the metrics table.
