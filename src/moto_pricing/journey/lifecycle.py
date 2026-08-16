"""
Renewal: age the book by a year, re-rate, and decide who stays.

Since 1 January 2022 the FCA's general insurance pricing practices rules
(GIPP, PS21/5) ban "price walking" — charging renewing customers more than
an equivalent new customer for the same risk. This engine deliberately
prices renewals with the *same* pricing model and the *same*
:class:`~moto_pricing.pricing.PricingAssumptions` used for new business.
The premium still moves year on year, but only because the risk did: NCD
step-up or step-down, and the rider/bike ageing a year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from burning_cost.data.aggregation import aggregate_claims

from moto_pricing.data.synthetic import generate_claims
from moto_pricing.pricing import PricingAssumptions, technical_to_gross

MAX_NCD_YEARS = 9
NCD_STEPS_LOST_PER_CLAIM = 2


@dataclass(frozen=True)
class RenewalAssumptions:
    base_retention_logit: float = 1.70   # ~85% baseline retention at an unchanged price
    price_elasticity: float = 2.6        # logit points lost per 100% renewal price increase
    claim_shopping_penalty: float = 0.45  # extra logit points lost if a claim was made this term
    idiosyncratic_noise_sd: float = 0.30


def _age_one_year(bound_policies: pd.DataFrame, experience: pd.DataFrame) -> pd.DataFrame:
    renewal = bound_policies.copy()
    renewal["rider_age"] += 1
    renewal["licence_years"] += 1
    renewal["bike_age"] += 1
    renewal["bike_value"] = (renewal["bike_value"] * 0.92).round(2)
    renewal["exposure"] = 1.0

    had_claim = experience["claim_count"] > 0
    ncd = renewal["ncd_years"].to_numpy()
    ncd = np.where(
        had_claim.to_numpy(),
        np.clip(ncd - NCD_STEPS_LOST_PER_CLAIM * experience["claim_count"].to_numpy(), 0, None),
        np.clip(ncd + 1, None, MAX_NCD_YEARS),
    )
    renewal["ncd_years"] = ncd.astype(int)
    return renewal


def simulate_renewals(
    bound_policies: pd.DataFrame,
    pricing_model,
    pricing_assumptions: PricingAssumptions | None = None,
    renewal_assumptions: RenewalAssumptions | None = None,
    claims_seed: int = 200,
    retention_seed: int = 201,
) -> pd.DataFrame:
    """
    Simulate first-term claims experience, then a renewal pricing and retention decision.

    Parameters
    ----------
    bound_policies : pd.DataFrame
        Bound new-business policies, e.g. the ``bound == True`` rows from
        :func:`moto_pricing.journey.conversion.simulate_new_business_conversion`.
        Must include ``payable_premium`` (the price they were sold at).
    pricing_model
        A fitted model exposing ``predict_pure_premium(df)``.
    pricing_assumptions : PricingAssumptions | None
        Applied identically to the renewal quote as to new business (no price walking).
    renewal_assumptions : RenewalAssumptions | None
    claims_seed, retention_seed : int

    Returns
    -------
    pd.DataFrame
        One row per bound policy: prior-term claim experience, the renewed
        rating factors, the renewal premium, the price change, and whether
        the policyholder renewed.
    """
    ra = renewal_assumptions or RenewalAssumptions()

    year_1_claims = generate_claims(bound_policies, seed=claims_seed)
    experience = aggregate_claims(bound_policies[["policy_id"]], year_1_claims, policy_key="policy_id")

    renewal_policies = _age_one_year(bound_policies, experience)
    renewal_risk_premium = pricing_model.predict_pure_premium(renewal_policies)
    renewal_priced = technical_to_gross(renewal_risk_premium, pricing_assumptions)

    result = renewal_policies.copy()
    result["prior_payable_premium"] = bound_policies["payable_premium"].to_numpy()
    result["year_1_claim_count"] = experience["claim_count"].to_numpy()
    result["year_1_incurred"] = experience["incurred"].to_numpy()
    result["renewal_risk_premium"] = renewal_priced["risk_premium"]
    result["renewal_payable_premium"] = renewal_priced["payable_premium"]
    result["renewal_price_change"] = (
        result["renewal_payable_premium"] / result["prior_payable_premium"] - 1.0
    )

    rng = np.random.default_rng(retention_seed)
    had_claim = (result["year_1_claim_count"] > 0).to_numpy()
    logit = (
        ra.base_retention_logit
        - ra.price_elasticity * result["renewal_price_change"].to_numpy()
        - ra.claim_shopping_penalty * had_claim
        + rng.normal(0, ra.idiosyncratic_noise_sd, len(result))
    )
    result["retention_probability"] = 1 / (1 + np.exp(-logit))
    result["renewed"] = rng.random(len(result)) < result["retention_probability"]
    return result
