"""
04_credibility.py
=================
Apply Bühlmann-Straub credibility to blend prior (GLM/EBM) predictions with
each policyholder's own claims experience.

This is especially useful at renewal: a rider with 3 years of claim history
deserves a personalised loading on top of the portfolio rate. Bühlmann-Straub
provides a statistically principled weight between the prior mean and the
individual experience mean.

Steps
-----
1. Build per-region claims history from training data.
2. Fit BuhlmannStraub model on region-level experience.
3. Print credibility factors Z by region.
4. Blend GLM pure premiums with region experience using credibility-weighted price.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import polars as pl
from insurance_credibility import BuhlmannStraub

from mc_pricing.data_generation import generate_full_portfolio


# ---------------------------------------------------------------------------
# Build per-region claims history
# ---------------------------------------------------------------------------

def build_region_history(df: pd.DataFrame) -> pl.DataFrame:
    """
    Aggregate training data to region × year level.

    BuhlmannStraub expects:
      group, period, loss_rate (= losses / exposure), weight (= exposure)
    """
    grouped = (
        df.groupby(["region", "calendar_year"], as_index=False)
        .agg(
            total_losses=("total_incurred", "sum"),
            total_exposure=("exposure", "sum"),
        )
        .rename(columns={"region": "group", "calendar_year": "period"})
    )
    grouped["loss_rate"] = grouped["total_losses"] / grouped["total_exposure"]
    return pl.from_pandas(grouped[["group", "period", "loss_rate", "total_exposure"]])


# ---------------------------------------------------------------------------
# Fit Bühlmann-Straub
# ---------------------------------------------------------------------------

def fit_buhlmann(history: pl.DataFrame) -> BuhlmannStraub:
    model = BuhlmannStraub()
    model.fit(history, group_col="group", period_col="period",
              loss_col="loss_rate", weight_col="total_exposure")
    return model


# ---------------------------------------------------------------------------
# Print credibility results
# ---------------------------------------------------------------------------

def print_results(model: BuhlmannStraub) -> None:
    premiums = model.premiums_   # Polars DataFrame: group, exposure, observed_mean, Z, credibility_premium
    print("\n=== Bühlmann-Straub Credibility Results ===")
    print(f"Grand mean (mu_hat):  {model.mu_hat_:,.2f}")
    print(f"Between-risk var (a): {model.a_hat_:.6f}")
    print(f"Within-risk var (v):  {model.v_hat_:.6f}")
    print(f"k = v/a:              {model.k_:.2f}\n")
    print(premiums.sort("Z", descending=True).to_pandas().to_string(index=False))


# ---------------------------------------------------------------------------
# Apply credibility to 2023 test-year GLM predictions
# ---------------------------------------------------------------------------

def apply_credibility_to_predictions(
    test: pd.DataFrame,
    model: BuhlmannStraub,
    glm_pp: np.ndarray,
) -> pd.DataFrame:
    """
    Credibility-blend each test policy's GLM pure premium with the
    region-level credibility premium.

    The credibility premium for each region is expressed as a loss rate
    (pure premium per unit exposure). We scale each policy's GLM prediction
    by (region credibility premium) / (portfolio mean).
    """
    premiums_df = model.premiums_.to_pandas()
    credibility_by_region = dict(
        zip(premiums_df["group"], premiums_df["credibility_premium"])
    )
    mu_hat = model.mu_hat_

    region_scale = {
        grp: cp / mu_hat
        for grp, cp in credibility_by_region.items()
    }

    result = test[["policy_id", "region", "exposure"]].copy()
    result["pp_glm"]         = glm_pp
    result["region_scale"]   = result["region"].map(region_scale).fillna(1.0)
    result["pp_credibility"] = result["pp_glm"] * result["region_scale"]
    result["actual_pp"]      = (
        test["claim_count"] * test["avg_severity"] / test["exposure"]
    ).values

    print("\n=== Credibility-Blended Pure Premiums by Region (2023) ===")
    summary = (
        result.groupby("region")[["pp_glm", "pp_credibility", "actual_pp"]]
        .mean()
        .round(2)
    )
    print(summary.to_string())

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> tuple[BuhlmannStraub, pd.DataFrame]:
    df = generate_full_portfolio()
    train = df[df["calendar_year"].isin([2021, 2022])].copy()
    test  = df[df["calendar_year"] == 2023].copy()

    history = build_region_history(train)
    model   = fit_buhlmann(history)
    print_results(model)

    # Flat prior: use portfolio mean frequency × mean severity as GLM proxy
    # (In a real pipeline, pass the GLM fitted values from 01_frequency_severity.py)
    overall_mean_pp = (
        (train["claim_count"] * train["avg_severity"]).sum() / train["exposure"].sum()
    )
    glm_pp_proxy = np.full(len(test), overall_mean_pp)

    result = apply_credibility_to_predictions(test, model, glm_pp_proxy)
    return model, result


if __name__ == "__main__":
    main()
