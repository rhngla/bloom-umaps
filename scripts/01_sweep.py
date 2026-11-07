#!/usr/bin/env python
"""Fit every UMAP in the sweep (cached), one fit per process; write results/sweep_manifest.csv.

python scripts/01_sweep.py --pilot     # one fit (the largest graph), for sizing --jobs
python scripts/01_sweep.py --jobs 24
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import umap
from joblib import Parallel, delayed

from bloom_umaps import io, sweep
from bloom_umaps.paths import FITS, RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("--pilot", action="store_true")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

if not FITS.parent.parent.resolve().as_posix().startswith("/opt/dlami/nvme/"):
    raise SystemExit(f"work/ must resolve to nvme, got {FITS.parent.parent.resolve()}")
X = io.read_embedding()
print(f"X {X.shape} {X.dtype}", flush=True)
g = sweep.grid()
if a.pilot:
    r = sweep.fit_one(X, 25, 0.2, 1.0, force=True)
    print("pilot", r, flush=True)
    raise SystemExit(0)

print(f"{len(g)} fits, {a.jobs} workers", flush=True)
t0 = time.time()
res = Parallel(n_jobs=a.jobs, verbose=5)(
    delayed(sweep.fit_one)(X, int(r.n_neighbors), float(r.min_dist), float(r.spread), a.force)
    for r in g.itertuples()
)
print(f"wall {time.time() - t0:.0f}s", flush=True)
man = g.merge(pd.DataFrame(res), on="fit_id")
man["metric"], man["random_state"] = sweep.METRIC, sweep.SEED
man["umap_learn"] = umap.__version__
man["input"], man["input_sha256"] = str(io.EMBEDDING), io.sha256(io.EMBEDDING)
RESULTS.mkdir(exist_ok=True)
prev = RESULTS / "sweep_manifest.csv"
if prev.exists():  # keep recorded fit times for fits that were cached this run
    old = pd.read_csv(prev).set_index("fit_id").seconds
    man["seconds"] = man.seconds.fillna(man.fit_id.map(old))
man.to_csv(prev, index=False)
print(man[["seconds"]].describe().T.to_string(), flush=True)
