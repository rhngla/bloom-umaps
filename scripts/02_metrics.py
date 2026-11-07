#!/usr/bin/env python
"""Seeded k-means NMI and within-subclass variance for every cached fit -> results/metrics.csv.

python scripts/02_metrics.py --jobs 24
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import sklearn
from joblib import Parallel, delayed

from bloom_umaps import io, metrics, sweep
from bloom_umaps.paths import RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=8)
a = ap.parse_args()

man = pd.read_csv(RESULTS / "sweep_manifest.csv")
labels = io.read_labels()
print(f"{len(man)} fits, {len(set(labels))} subclasses, {len(labels)} cells", flush=True)


def one(fid: str) -> dict:
    return dict(fit_id=fid, **metrics.all_metrics(sweep.read_fit(fid), labels))


t0 = time.time()
rows = Parallel(n_jobs=a.jobs, verbose=5)(delayed(one)(f) for f in man.fit_id)
print(f"wall {time.time() - t0:.0f}s", flush=True)
out = man[["fit_id", "n_neighbors", "min_dist", "spread"]].merge(pd.DataFrame(rows), on="fit_id")
out["k"], out["sklearn"] = len(set(labels)), sklearn.__version__
out.to_csv(RESULTS / "metrics.csv", index=False, float_format="%.6g")
print(out[["nmi_mean", "nmi_sd", "v_span", "v_total"]].describe().to_string(), flush=True)
