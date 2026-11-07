"""Checks on docs/data, skipped until scripts/04_export_dashboard.py has run."""

import json
from pathlib import Path

import numpy as np
import pytest

from bloom_umaps import sweep

DATA = Path(__file__).resolve().parents[1] / "docs" / "data"
pytestmark = pytest.mark.skipif(not (DATA / "metrics.json").exists(), reason="no export yet")


def test_counts_and_tables():
    fits = json.loads((DATA / "metrics.json").read_text())["fits"]
    cells = json.loads((DATA / "cells.json").read_text())
    assert len(fits) == 100 and cells["n"] == 15_750
    assert len(cells["subclass"]) == len(cells["klass"]) == cells["n"]
    assert len(cells["subclasses"]) == 315
    assert len(set(cells["klass"])) == 41 and max(cells["klass"]) < len(cells["classes"])
    assert max(cells["subclass"]) < 315
    assert len(cells["subclass_slot"]) == 315 and len(cells["subclass_slot_colors"]) <= 256
    assert max(cells["subclass_slot"]) < len(cells["subclass_slot_colors"])
    assert len(cells["classes"]) <= 256


def test_decoded_coordinates_match_fits():
    fits = json.loads((DATA / "metrics.json").read_text())["fits"]
    if not sweep.FITS.exists():
        pytest.skip("fits not on this machine")
    for f in fits:
        q = np.fromfile(DATA / "fits" / f"{f['fit_id']}.bin", dtype="<u2").astype("float64")
        n = q.size // 2
        xy = np.stack([f["x0"] + q[:n] * f["scale"], f["y0"] + q[n:] * f["scale"]], axis=1)
        err = np.abs(xy - sweep.read_fit(f["fit_id"])).max()
        assert err <= 0.5 * f["scale"] * (1 + 1e-6), (f["fit_id"], err, f["scale"])
