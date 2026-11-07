#!/usr/bin/env python
"""Export the static dashboard's data into docs/data (served by GitHub Pages from main:/docs).

    python scripts/04_export_dashboard.py

Writes
    docs/data/metrics.json      one record per fit: params, NMI, variances, Pareto flags,
                                and the dequantisation + display-frame parameters
    docs/data/cells.json        per-cell subclass / class index and the name/colour tables
    docs/data/fits/<id>.bin     little-endian uint16, x[0..n) then y[0..n), one scale for
                                both axes: x = x0 + q * scale (no original embedding)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from bloom_umaps import colours, io, metrics, plots, sweep
from bloom_umaps.paths import DATA, REPO, RESULTS

OUT = REPO / "docs" / "data"
QMAX = 65535

m = pd.read_csv(RESULTS / "metrics.csv")
m["pareto_v_span"] = plots.pareto_mask(m.v_span.to_numpy(), m.nmi_mean.to_numpy())
m["pareto_v_total"] = plots.pareto_mask(m.v_total.to_numpy(), m.nmi_mean.to_numpy())

(OUT / "fits").mkdir(parents=True, exist_ok=True)
recs, max_err = [], 0.0
for r in m.itertuples():
    Y = sweep.read_fit(r.fit_id).astype("float64")
    lo = Y.min(axis=0)
    scale = float((Y.max(axis=0) - lo).max() / QMAX)  # isotropic: same step on x and y
    q = np.rint((Y - lo) / scale).astype("<u2")
    max_err = max(max_err, float(np.abs(lo + q * scale - Y).max() / scale))
    np.concatenate([q[:, 0], q[:, 1]]).tofile(OUT / "fits" / f"{r.fit_id}.bin")
    flo, fhi = np.percentile(Y, metrics.FRAME_Q, axis=0)
    recs.append(
        dict(
            fit_id=r.fit_id,
            n_neighbors=int(r.n_neighbors),
            min_dist=float(r.min_dist),
            spread=float(r.spread),
            nmi_mean=round(r.nmi_mean, 5),
            nmi_sd=round(r.nmi_sd, 5),
            v_span=round(r.v_span, 6),
            v_total=round(r.v_total, 5),
            pareto_v_span=bool(r.pareto_v_span),
            pareto_v_total=bool(r.pareto_v_total),
            x0=float(lo[0]),
            y0=float(lo[1]),
            scale=scale,
            frame_cx=float((flo[0] + fhi[0]) / 2),
            frame_cy=float((flo[1] + fhi[1]) / 2),
            frame_span=metrics.frame_span(Y),
        )
    )
if max_err > 0.5 + 1e-9:
    raise SystemExit(f"quantisation error {max_err:.3f} steps > 0.5")

sub = io.read_labels(col="AIT33_subclass")
cls = io.read_labels(col="AIT33_class")
tax = pd.read_csv(DATA / "ait33_subclass_taxonomy.csv")
tax = tax[tax.subclass_id_label.isin(set(sub))].sort_values("subclass_id_label")
if len(tax) != len(set(sub)):
    raise SystemExit(f"{len(set(sub)) - len(tax)} subclasses missing from taxonomy")
# Classes: every class used by a cell's own AIT33_class OR as a subclass's parent; colours
# from the full taxonomy. metadata.h5ad assigns class and subclass per cell independently, so
# a few cells carry a class that is not their subclass's parent; the page colours by the
# cell's own class and flags those cells in the tooltip.
all_cls = pd.read_csv(DATA / "ait33_subclass_taxonomy.csv")[["class_id_label", "class_color"]]
ccol = dict(all_cls.drop_duplicates().itertuples(index=False, name=None))
names = sorted(set(cls) | set(tax.class_id_label))
missing = [c for c in names if c not in ccol]
if missing:
    raise SystemExit(f"classes without a taxonomy colour: {missing}")
classes = pd.DataFrame(dict(class_id_label=names, class_color=[ccol[c] for c in names]))
s_idx = {s: i for i, s in enumerate(tax.subclass_id_label)}
c_idx = {c: i for i, c in enumerate(classes.class_id_label)}
pair = dict(zip(tax.subclass_id_label, tax.class_id_label, strict=True))
bad = sum(pair[s] != c for s, c in zip(sub, cls, strict=True))
print(f"{bad} cells whose AIT33_class is not their subclass's parent class (flagged in tooltip)")
slot_of, slot_colours, pairs = colours.merge_to_slots(
    tax.subclass_color.tolist(), tax.class_id_label.tolist(), max_slots=256
)
names_ = tax.subclass_id_label.tolist()
pd.DataFrame(
    [
        dict(
            subclass_a=names_[i],
            subclass_b=names_[j],
            delta_e=round(de, 2),
            colour=tax.subclass_color.iloc[i],
        )
        for i, j, de in pairs
    ]
).to_csv(REPO / "results" / "dashboard_subclass_colour_merges.csv", index=False)
print(
    f"subclass colours: {len(names_)} subclasses -> {len(slot_colours)} slots, "
    f"{len(pairs)} merged pairs, dE median {np.median([p[2] for p in pairs]):.1f}, "
    f"max {max(p[2] for p in pairs):.1f}, same-class pairs "
    f"{sum(tax.class_id_label.iloc[i] == tax.class_id_label.iloc[j] for i, j, _ in pairs)}"
)
cells = dict(
    subclass_slot=slot_of,
    subclass_slot_colors=slot_colours,
    n=len(sub),
    subclass=[s_idx[s] for s in sub],
    klass=[c_idx[c] for c in cls],
    subclasses=[
        dict(name=s, color=col, klass=c_idx[k])
        for s, col, k in zip(
            tax.subclass_id_label, tax.subclass_color, tax.class_id_label, strict=True
        )
    ],
    classes=[
        dict(name=c, color=col)
        for c, col in zip(classes.class_id_label, classes.class_color, strict=True)
    ],
)
meta = dict(
    dataset="allen_multiome eval_balanced_50 (15,750 cells)",
    embedding="Bloom slurm_3458_step_530000, RNA top4096 (1024-d), used as stored",
    umap="umap-learn 0.5.12, metric=euclidean, random_state=0",
    nmi="seeded k-means (K = 315 subclasses, one random cell per subclass), mean of seeds 0-2",
    v_span="mean within-subclass variance / displayed span^2",
    v_total="mean within-subclass variance / total variance",
    n_cells=len(sub),
    n_fits=len(recs),
)
(OUT / "metrics.json").write_text(json.dumps(dict(meta=meta, fits=recs), separators=(",", ":")))
(OUT / "cells.json").write_text(json.dumps(cells, separators=(",", ":")))
size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
print(
    f"{len(recs)} fits, {len(sub)} cells, {len(tax)} subclasses, {len(classes)} classes "
    f"({len(set(cls))} used as a cell's own class)"
)
print(f"max quantisation error {max_err:.3f} steps; docs/data {size / 1e6:.1f} MB")
