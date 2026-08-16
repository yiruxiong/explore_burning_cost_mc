#!/usr/bin/env python3
"""CLI entry point for the full pricing pipeline.

Usage
-----
    uv run scripts/run_pipeline.py --n-policies 150000 --n-prospects 40000
"""

from __future__ import annotations

import click

from moto_pricing.config import OUTPUTS_DIR, RANDOM_SEED
from moto_pricing.pipeline import run_full_pipeline


@click.command()
@click.option("--n-policies", default=150_000, show_default=True, help="Historical book size for training.")
@click.option("--n-prospects", default=40_000, show_default=True, help="New-business quote cohort size.")
@click.option("--seed", default=RANDOM_SEED, show_default=True, help="Base random seed.")
@click.option("--outputs-dir", default=str(OUTPUTS_DIR), show_default=True, help="Where to write the report.")
def main(n_policies: int, n_prospects: int, seed: int, outputs_dir: str) -> None:
    from pathlib import Path

    result = run_full_pipeline(
        n_policies=n_policies, n_prospects=n_prospects, seed=seed, outputs_dir=Path(outputs_dir),
    )
    click.echo(result.model_comparison.to_string(index=False))
    click.echo(f"\nReport written to {outputs_dir}")


if __name__ == "__main__":
    main()
