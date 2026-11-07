"""Fit 315 subclass colours into EmbeddingView's 256 Uint8 categories by merging close pairs."""

from __future__ import annotations

import numpy as np


def hex_to_lab(hexes: list[str]) -> np.ndarray:
    """sRGB hex -> CIELAB (D65), (n, 3)."""
    rgb = np.array([[int(h.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)] for h in hexes])
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def merge_to_slots(colours: list[str], groups: list, max_slots: int = 256):
    """Greedy: merge the closest-colour pairs (CIE76 dE) until n <= max_slots.

    Each item joins at most one pair; pairs within the same group (class) are only used
    after every cross-group candidate is exhausted. Returns (slot_of_item, slot_colours,
    pairs) where a merged slot takes the colour of the pair's first (lower-index) item.
    """
    n = len(colours)
    need = max(0, n - max_slots)
    lab = hex_to_lab(colours)
    d = np.sqrt(((lab[:, None] - lab[None]) ** 2).sum(-1))
    g = np.asarray(groups)
    iu, ju = np.triu_indices(n, 1)
    same = g[iu] == g[ju]
    order = np.lexsort((d[iu, ju], same))  # cross-group first, then by distance
    used = np.zeros(n, bool)
    pairs = []
    for k in order:
        if len(pairs) == need:
            break
        i, j = iu[k], ju[k]
        if not (used[i] or used[j]):
            used[i] = used[j] = True
            pairs.append((int(i), int(j), float(d[i, j])))
    if len(pairs) < need:
        raise ValueError(f"could only form {len(pairs)} of {need} pairs")
    partner = {j: i for i, j, _ in pairs}
    slot_of, slot_colours, slot_index = [], [], {}
    for i in range(n):
        head = partner.get(i, i)
        if head not in slot_index:
            slot_index[head] = len(slot_colours)
            slot_colours.append(colours[head])
        slot_of.append(slot_index[head])
    return slot_of, slot_colours, pairs
