"""NMI vs within-subclass variance across the sweep, and thumbnails of selected fits."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

MM = 1 / 25.4
SPREAD_MARKER = {1.0: "o", 1.5: "s", 2.0: "^"}


def style() -> None:
    """Three-size font ladder (8/7/6 pt), outward ticks, frameless legends, 300 dpi."""
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )


def nn_colours(levels) -> dict:
    """Ordinal single-family ramp for n_neighbors."""
    cmap = plt.get_cmap("viridis")
    return {v: cmap(x) for v, x in zip(levels, np.linspace(0.05, 0.85, len(levels)), strict=True)}


def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True where no other point has x >= and y >= with at least one strictly greater."""
    m = np.ones(len(x), bool)
    for i in range(len(x)):
        dom = (x >= x[i]) & (y >= y[i]) & ((x > x[i]) | (y > y[i]))
        m[i] = not dom.any()
    return m


def frame_umap(ax, Y: np.ndarray, pad: float = 1.06, q=(0.2, 99.8)) -> None:
    """Bare square frame with the same span on both axes (as bloom-figures umap_frame)."""
    lo, hi = np.percentile(Y, q, axis=0)
    span = pad * np.max(hi - lo)
    c = (lo + hi) / 2
    ax.set_xlim(c[0] - span / 2, c[0] + span / 2)
    ax.set_ylim(c[1] - span / 2, c[1] + span / 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def nmi_vs_variance(m: pd.DataFrame, out, xcol: str = "v_span") -> list[str]:
    """a: every fit on one axes; b-d: one panel per spread, lines trace increasing min_dist.

    Returns the fit_ids on the Pareto front (max NMI and max variance).
    """
    style()
    m = m.sort_values(["spread", "n_neighbors", "min_dist"]).reset_index(drop=True)
    x, y = m[xcol].to_numpy(), m.nmi_mean.to_numpy()
    front = pareto_mask(x, y)
    nns = sorted(m.n_neighbors.unique())
    col = nn_colours(nns)
    xlab = {
        "v_span": "Within-subclass variance / displayed span\u00b2",
        "v_total": "Within-subclass variance / total variance",
    }[xcol]

    fig = plt.figure(figsize=(180 * MM, 120 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.5, wspace=0.08)
    ax = fig.add_subplot(gs[0, :])
    for i, r in m.iterrows():
        ax.errorbar(
            r[xcol], r.nmi_mean, yerr=r.nmi_sd, fmt="none", ecolor="0.6", elinewidth=0.6, zorder=1
        )
        ax.scatter(
            r[xcol],
            r.nmi_mean,
            s=22,
            marker=SPREAD_MARKER[r.spread],
            color=col[r.n_neighbors],
            edgecolor="k" if front[i] else "none",
            linewidth=0.9,
            zorder=3,
        )
    order = np.flatnonzero(front)[np.argsort(x[front])]
    for k, i in enumerate(order, 1):
        up = k % 2 == 1  # alternate above / below so near-coincident neighbours stay legible
        ax.annotate(
            str(k),
            (x[i], y[i]),
            xytext=(0, 5 if up else -6),
            textcoords="offset points",
            ha="center",
            va="bottom" if up else "top",
            fontsize=6,
            color="0.15",
        )
    ax.set_xlabel(xlab)
    ax.set_ylabel("Seeded k-means NMI\n(mean \u00b1 sd, 3 seeds)")
    ax.set_title(
        f"All {len(m)} UMAP fits; outlined and numbered = Pareto front "
        "(no other fit has both higher NMI and higher variance)",
        loc="left",
    )
    ax.margins(0.04)
    h = [
        Line2D([], [], ls="none", marker="o", ms=5, color=col[v], label=f"n_neighbors {v}")
        for v in nns
    ]
    h += [
        Line2D([], [], ls="none", marker=mk, ms=5, color="0.4", label=f"spread {sp:g}")
        for sp, mk in SPREAD_MARKER.items()
    ]
    ax.legend(
        handles=h,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        handletextpad=0.3,
        borderaxespad=0,
    )

    axs = [fig.add_subplot(gs[1, j]) for j in range(3)]
    for j, (sp, g) in enumerate(m.groupby("spread")):
        a = axs[j]
        for nn, gg in g.groupby("n_neighbors"):
            gg = gg.sort_values("min_dist")
            a.plot(
                gg[xcol], gg.nmi_mean, "-", color=col[nn], lw=0.9, marker=SPREAD_MARKER[sp], ms=3
            )
            a.plot(
                gg[xcol].iloc[0],
                gg.nmi_mean.iloc[0],
                ls="none",
                marker=SPREAD_MARKER[sp],
                ms=6,
                mfc=col[nn],
                mec="k",
                mew=0.8,
            )
        a.set_title(f"spread {sp:g} (min_dist 0.2\u2013{g.min_dist.max():g})", loc="left")
        a.set_xlim(ax.get_xlim()), a.set_ylim(ax.get_ylim())
        a.set_xlabel(xlab if j == 1 else "")
        if j == 0:
            a.set_ylabel("Seeded k-means NMI")
        else:
            a.tick_params(labelleft=False)
    axs[-1].legend(
        handles=[
            Line2D(
                [],
                [],
                color="0.4",
                lw=0.9,
                marker="^",
                ms=3,
                label="one n_neighbors, min_dist increasing",
            ),
            Line2D(
                [],
                [],
                ls="none",
                marker="^",
                ms=6,
                mfc="0.6",
                mec="k",
                mew=0.8,
                label="min_dist 0.2 (line start)",
            ),
        ],
        loc="lower left",
        fontsize=6,
        handletextpad=0.4,
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return m.loc[order, "fit_id"].tolist()


def thumbnails(
    fits: list[tuple[str, np.ndarray, str]], labels: np.ndarray, colours: dict, out, ncol: int = 4
) -> None:
    """Grid of UMAP thumbnails coloured by subclass; fits = [(fit_id, Y, caption)]."""
    style()
    nrow = int(np.ceil(len(fits) / ncol))
    fig, axs = plt.subplots(
        nrow,
        ncol,
        figsize=(45 * ncol * MM, 49 * nrow * MM),
        squeeze=False,
        gridspec_kw=dict(wspace=0.05, hspace=0.25),
    )
    c = np.array([colours.get(lab, "#999999") for lab in labels])
    perm = np.random.default_rng(0).permutation(len(labels))
    for a in axs.ravel():
        a.set_axis_off()
    for a, (_fid, Y, cap) in zip(axs.ravel(), fits, strict=False):
        a.set_axis_on()
        a.scatter(
            Y[perm, 0], Y[perm, 1], s=0.3, c=c[perm], linewidths=0, alpha=0.6, rasterized=True
        )
        frame_umap(a, Y)
        a.set_title(cap, fontsize=6)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
