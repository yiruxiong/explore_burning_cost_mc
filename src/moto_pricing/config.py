"""Shared constants for the pricing pipeline."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"

# UK Insurance Premium Tax, standard rate.
IPT_RATE = 0.12

RANDOM_SEED = 42
