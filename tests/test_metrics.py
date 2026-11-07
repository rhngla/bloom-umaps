import numpy as np

from bloom_umaps import metrics, sweep


def blobs(sep=10.0, sd=1.0, k=6, n=50, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-sep * k, sep * k, size=(k, 2))
    Y = np.vstack([c + rng.normal(0, sd, size=(n, 2)) for c in centres])
    return Y, np.repeat([f"c{i}" for i in range(k)], n)


def test_grid_has_100_valid_fits():
    g = sweep.grid()
    assert len(g) == 100 and (g.min_dist <= g.spread + 1e-9).all()
    assert g.fit_id.is_unique


def test_seeded_init_one_cell_per_class():
    Y, lab = blobs()
    init = metrics.seeded_init(Y, lab, seed=1)
    assert init.shape == (6, 2)
    for c, row in zip(np.unique(lab), init, strict=True):
        assert any(np.allclose(row, y) for y in Y[lab == c])


def test_nmi_separated_blobs_is_one():
    Y, lab = blobs(sep=50, sd=0.5)
    assert metrics.seeded_kmeans_nmi(Y, lab, seed=0) > 0.999


def test_variance_scale_and_shift_invariant():
    Y, lab = blobs()
    a = metrics.variance_metrics(Y, lab)
    b = metrics.variance_metrics(7.3 * Y + 4.0, lab)
    assert np.isclose(a["v_span"], b["v_span"]) and np.isclose(a["v_total"], b["v_total"])


def test_variance_collapse_goes_to_zero():
    Y, lab = blobs(sd=1.0)
    tight, _ = blobs(sd=1e-4)
    assert metrics.variance_metrics(tight, lab)["v_span"] < 1e-6
    assert metrics.variance_metrics(Y, lab)["v_span"] > 1e-4


def test_v_total_bounded():
    Y, lab = blobs(sd=5.0)
    assert 0 < metrics.variance_metrics(Y, lab)["v_total"] <= 1
