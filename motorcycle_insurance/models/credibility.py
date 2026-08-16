"""
Credibility weighting for thin-data rating factor segments.

Uses Bühlmann-Straub credibility to blend A/E ratios for regions and
Poisson-Gamma conjugate credibility for occupation segments.  Both models
blend observed experience with the portfolio mean in proportion to the
volume of evidence available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from insurance_credibility import BuhlmannStraub, PoissonGammaCredibility


def fit_region_credibility(
    train: pd.DataFrame,
    glm_freq_predictions: np.ndarray,
) -> dict:
    """Fit Bühlmann-Straub credibility by region and policy year.

    Derives credibility-weighted A/E frequency relativities for each region
    using multi-year Bühlmann-Straub.

    Parameters
    ----------
    train:
        Training data with ``claim_count``, ``exposure``, ``region``,
        and ``policy_year`` columns.
    glm_freq_predictions:
        GLM expected claim counts aligned to ``train``.

    Returns
    -------
    dict
        ``model``    — fitted :class:`~insurance_credibility.BuhlmannStraub`
        ``regions``  — list of region names seen in training data
        ``cred_rel`` — dict mapping region → credibility A/E relativity
    """
    df = train.copy()
    df["glm_expected"] = glm_freq_predictions

    # Annual A/E ratios at region × policy_year level
    year_region = (
        df.groupby(["region", "policy_year"])
        .agg(
            total_claims=("claim_count", "sum"),
            total_expected=("glm_expected", "sum"),
        )
        .reset_index()
    )
    year_region["ae_ratio"] = (
        year_region["total_claims"]
        / year_region["total_expected"].clip(lower=1e-6)
    )

    bs_data = pd.DataFrame(
        {
            "group": year_region["region"],
            "period": year_region["policy_year"],
            "loss": year_region["ae_ratio"],
            "weight": year_region["total_expected"],
        }
    )

    bs = BuhlmannStraub()
    bs.fit(bs_data)

    # premiums_ is a Polars DataFrame; convert to dict
    premiums_df = bs.premiums_.to_pandas()
    cred_rel = dict(
        zip(
            premiums_df["group"].tolist(),
            premiums_df["credibility_premium"].tolist(),
        )
    )

    return {
        "model": bs,
        "regions": premiums_df["group"].tolist(),
        "cred_rel": cred_rel,
    }


def fit_occupation_credibility(
    train: pd.DataFrame,
    glm_freq_predictions: np.ndarray,
) -> dict:
    """Fit Poisson-Gamma conjugate credibility by occupation.

    Parameters
    ----------
    train:
        Training data with ``claim_count``, ``occupation``, ``exposure`` columns.
    glm_freq_predictions:
        GLM expected claim counts aligned to ``train``.

    Returns
    -------
    dict
        ``model``    — fitted :class:`~insurance_credibility.PoissonGammaCredibility`
        ``cred_rel`` — dict mapping occupation → posterior credibility rate
        ``prior_mean`` — portfolio prior mean frequency
    """
    df = train.copy()
    df["expected"] = glm_freq_predictions

    occ_stats = (
        df.groupby("occupation")
        .agg(
            total_claims=("claim_count", "sum"),
            total_expected=("expected", "sum"),
        )
        .reset_index()
    )

    pgc_data = pd.DataFrame(
        {
            "group": occ_stats["occupation"],
            "claims": occ_stats["total_claims"].astype(float),
            "exposure": occ_stats["total_expected"],
        }
    )

    pgc = PoissonGammaCredibility()
    pgc.fit(pgc_data, group_col="group", claims_col="claims", exposure_col="exposure")

    premiums_df = pgc.premiums_.to_pandas()
    # credibility_rate is the posterior blended frequency rate
    # Express as relativity relative to the portfolio prior mean
    prior_mean = pgc.prior_mean_
    cred_rel = {
        row["group"]: row["credibility_rate"] / prior_mean if prior_mean > 0 else 1.0
        for _, row in premiums_df.iterrows()
    }

    return {
        "model": pgc,
        "cred_rel": cred_rel,
        "prior_mean": prior_mean,
    }


def apply_credibility_adjustment(
    data: pd.DataFrame,
    base_predictions: np.ndarray,
    region_cred: dict,
    occupation_cred: dict,
) -> np.ndarray:
    """Apply region and occupation credibility relativities to base predictions.

    Parameters
    ----------
    data:
        Portfolio data with ``region`` and ``occupation`` columns.
    base_predictions:
        GLM or GAM pure-premium predictions.
    region_cred:
        Output of :func:`fit_region_credibility`.
    occupation_cred:
        Output of :func:`fit_occupation_credibility`.

    Returns
    -------
    np.ndarray of credibility-adjusted pure premiums.
    """
    region_rel = np.array(
        [region_cred["cred_rel"].get(r, 1.0) for r in data["region"]]
    )
    occ_rel = np.array(
        [occupation_cred["cred_rel"].get(o, 1.0) for o in data["occupation"]]
    )
    return base_predictions * region_rel * occ_rel
