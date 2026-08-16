"""
Assemble the policy-level modelling dataset from raw synthetic tables.

This is the one place the pipeline calls into ``burning_cost`` directly:
``calculate_exposure`` turns policy inception/expiry dates into an earned
duration, and ``aggregate_claims`` joins the separate claims table onto the
policy table. Both are the kind of fiddly, easy-to-get-wrong plumbing that
``burning_cost`` exists to handle correctly (leap years, mid-term
cancellations, policies with zero claims).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from burning_cost.data.aggregation import aggregate_claims
from burning_cost.data.exposure import calculate_exposure

from moto_pricing.data.synthetic import generate_claims, generate_policies


@dataclass(frozen=True)
class ModellingDataset:
    """The synthetic book, in the shapes each pipeline stage needs.

    Attributes
    ----------
    policies : pd.DataFrame
        Raw policy table, one row per policy.
    claims : pd.DataFrame
        Raw claims table, one row per claim.
    modelling : pd.DataFrame
        One row per policy: rating factors plus ``exposure`` (written basis,
        i.e. the full earned duration of the policy term), ``claim_count``,
        ``incurred`` and ``avg_severity``. This is what the frequency and
        severity models train on.
    exposure_by_accident_year : pd.Series
        Earned exposure (calendar-accurate, split across year boundaries)
        summed by accident year — a standard experience-monitoring cut that
        the policy-level ``modelling`` frame cannot produce on its own.
    """

    policies: pd.DataFrame
    claims: pd.DataFrame
    modelling: pd.DataFrame
    exposure_by_accident_year: pd.Series


def build_modelling_dataset(n_policies: int, seed: int = 42) -> ModellingDataset:
    """
    Generate a synthetic motorcycle book and assemble it into modelling shape.

    Parameters
    ----------
    n_policies : int
        Number of policies to generate.
    seed : int
        Random seed for the policy table; the claims table uses ``seed + 1``.

    Returns
    -------
    ModellingDataset
    """
    policies = generate_policies(n_policies, seed=seed)
    claims = generate_claims(policies, seed=seed + 1)

    written_exposure = calculate_exposure(policies, basis="written")
    modelling = aggregate_claims(written_exposure, claims, policy_key="policy_id")

    earned_exposure = calculate_exposure(policies, basis="earned")
    exposure_by_accident_year = (
        earned_exposure.groupby("accident_year")["exposure"].sum().sort_index()
    )

    return ModellingDataset(
        policies=policies,
        claims=claims,
        modelling=modelling,
        exposure_by_accident_year=exposure_by_accident_year,
    )
