#!/usr/bin/env python
"""results/nmi_vs_variance.png (+ _vtotal variant) and thumbnails of the Pareto-front fits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from bloom_umaps import io, plots, sweep
from bloom_umaps.paths import DATA, RESULTS

m = pd.read_csv(RESULTS / "metrics.csv")
front = plots.nmi_vs_variance(m, RESULTS / "nmi_vs_variance.png", "v_span")
front_t = plots.nmi_vs_variance(m, RESULTS / "nmi_vs_variance_vtotal.png", "v_total")
print("pareto (v_span): ", front)
print("pareto (v_total):", front_t)

tax = pd.read_csv(DATA / "ait33_subclass_taxonomy.csv")
colours = dict(zip(tax.subclass_id_label, tax.subclass_color, strict=False))
labels = io.read_labels()
f = m.set_index("fit_id").loc[front]  # already ordered by v_span, as numbered in the plot
fits = [
    (
        fid,
        sweep.read_fit(fid),
        f"{k}: nn {r.n_neighbors}, md {r.min_dist:g}, sp {r.spread:g}\n"
        f"NMI {r.nmi_mean:.3f}, V {r.v_span:.4f}",
    )
    for k, (fid, r) in enumerate(f.iterrows(), 1)
]
plots.thumbnails(fits, labels, colours, RESULTS / "pareto_thumbnails.png")
print("wrote", RESULTS / "pareto_thumbnails.png")
