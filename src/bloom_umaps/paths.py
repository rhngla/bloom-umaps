"""Repo-relative paths. Stored embeddings go under WORK (nvme symlink) only."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work"  # -> /opt/dlami/nvme/work/bloom-umaps (gitignored symlink)
FITS = WORK / "sweep" / "fits"
RESULTS = REPO / "results"
DATA = REPO / "data"
