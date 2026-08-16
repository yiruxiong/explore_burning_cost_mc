"""
Synthetic UK motorcycle insurance portfolio generator.

Generates policy-level data with realistic UK market distributions for
rating factors, claim counts (Poisson), claim severities (log-normal),
and a simple price-sensitivity conversion curve.  The data is large enough
(50 000 policies by default) to support credibility graduation, GAM fitting,
monitoring splits and ensemble training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 42


def generate_portfolio(n_policies: int = 50_000, seed: int = RNG_SEED) -> pd.DataFrame:
    """Return a DataFrame of synthetic UK motorcycle insurance policies.

    Each row represents one annual policy period.  The synthetic data
    follows realistic UK market distributions for motorcycle insurance.

    Parameters
    ----------
    n_policies:
        Number of policy records to generate.
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns
        -------
        policy_id       : unique policy identifier
        age             : rider age (years)
        years_licensed  : years since full motorcycle licence
        bike_cc         : engine displacement (cc)
        bike_age        : age of the motorcycle (years)
        bike_value      : estimated market value of the motorcycle (£)
        annual_mileage  : estimated annual mileage
        region          : UK region (one of 11 regions)
        occupation      : simplified occupation category
        ncb             : no-claims bonus steps (0–5)
        gender          : M / F (used for fairness audit only — not a rating factor)
        exposure        : policy exposure in years (0.1–1.0)
        claim_count     : number of claims during the period
        avg_severity    : average claim cost (£), NaN when claim_count == 0
        total_loss      : total incurred loss (£)
        base_rate       : the "true" technical premium used as ground truth
        quoted_premium  : price quoted to the customer
        converted       : whether the customer bought (1 = yes, 0 = no)
        policy_year     : calendar year of policy inception (2018–2023)
    """
    rng = np.random.default_rng(seed)

    # ── Rating factors ────────────────────────────────────────────────────────
    age = rng.choice(
        np.arange(17, 80),
        size=n_policies,
        p=_age_weights(np.arange(17, 80)),
    ).astype(float)

    years_licensed = np.clip(
        rng.exponential(scale=8, size=n_policies), 0, age - 17
    ).round().astype(float)

    bike_cc = rng.choice(
        [125, 250, 400, 600, 650, 750, 900, 1000, 1100, 1200, 1300],
        size=n_policies,
        p=[0.18, 0.08, 0.07, 0.12, 0.10, 0.09, 0.08, 0.12, 0.06, 0.05, 0.05],
    ).astype(float)

    bike_age = rng.choice(np.arange(0, 21), size=n_policies).astype(float)

    bike_value = np.exp(
        rng.normal(loc=8.5, scale=0.8, size=n_policies)
    ).clip(500, 30_000)

    annual_mileage = np.exp(
        rng.normal(loc=8.3, scale=0.5, size=n_policies)
    ).clip(500, 30_000).round(-2)

    regions = [
        "London", "South East", "South West", "East of England",
        "West Midlands", "East Midlands", "Yorkshire", "North West",
        "North East", "Scotland", "Wales",
    ]
    region_probs = [0.14, 0.13, 0.08, 0.09, 0.10, 0.08, 0.09, 0.10, 0.05, 0.08, 0.06]
    region = rng.choice(regions, size=n_policies, p=region_probs)

    occupations = ["Professional", "Manual", "Student", "Retired", "Self-employed", "Other"]
    occupation_probs = [0.30, 0.25, 0.15, 0.10, 0.12, 0.08]
    occupation = rng.choice(occupations, size=n_policies, p=occupation_probs)

    ncb = rng.choice(np.arange(0, 6), size=n_policies,
                     p=[0.20, 0.15, 0.15, 0.15, 0.15, 0.20])

    gender = rng.choice(["M", "F"], size=n_policies, p=[0.82, 0.18])

    exposure = rng.uniform(0.1, 1.0, size=n_policies).round(3)

    policy_year = rng.choice(np.arange(2018, 2024), size=n_policies,
                             p=[0.12, 0.14, 0.16, 0.18, 0.20, 0.20])

    # ── True log-rate (ground truth technical rate per unit exposure) ─────────
    log_rate = (
        -2.8                                              # intercept
        + _age_effect(age)
        + 0.04 * np.log1p(bike_cc / 100)                 # larger engine → more claims
        - 0.02 * np.log1p(bike_age)                      # older bikes → lower frequency (newer is riskier)
        + 0.10 * np.log1p(annual_mileage / 1000)         # more miles → more claims
        - 0.05 * ncb                                     # NCB discount
        + _region_effect(region)
        + _occupation_effect(occupation)
        + rng.normal(0, 0.05, size=n_policies)           # residual heterogeneity
    )
    true_freq = np.exp(log_rate) * exposure

    # ── Claim frequency ───────────────────────────────────────────────────────
    claim_count = rng.poisson(true_freq)

    # ── Claim severity ────────────────────────────────────────────────────────
    log_sev_mean = (
        6.5
        + 0.30 * np.log1p(bike_value / 5000)
        + 0.15 * np.log1p(bike_cc / 600)
        + _region_severity_effect(region)
    )
    avg_severity = np.where(
        claim_count > 0,
        np.exp(rng.normal(log_sev_mean, 0.5, size=n_policies)),
        np.nan,
    )
    total_loss = np.where(claim_count > 0, claim_count * avg_severity, 0.0)

    # ── Base rate & quoted premium ────────────────────────────────────────────
    true_severity_mean = np.exp(log_sev_mean + 0.5**2 / 2)
    base_rate = (np.exp(log_rate) * true_severity_mean).round(2)

    # Quoted premium: base rate + loading + some noise
    loading_factor = rng.uniform(0.85, 1.25, size=n_policies)
    quoted_premium = (base_rate * loading_factor).round(2)

    # ── Conversion (price-to-sell) ────────────────────────────────────────────
    # Elasticity: probability of buying decreases as quote / base_rate rises
    price_ratio = quoted_premium / base_rate
    buy_prob = _conversion_curve(price_ratio, rng)
    converted = rng.binomial(1, buy_prob)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "policy_id": np.arange(1, n_policies + 1),
            "age": age.astype(int),
            "years_licensed": years_licensed.astype(int),
            "bike_cc": bike_cc.astype(int),
            "bike_age": bike_age.astype(int),
            "bike_value": bike_value.round(0).astype(int),
            "annual_mileage": annual_mileage.astype(int),
            "region": region,
            "occupation": occupation,
            "ncb": ncb.astype(int),
            "gender": gender,
            "exposure": exposure,
            "claim_count": claim_count,
            "avg_severity": avg_severity.round(2),
            "total_loss": total_loss.round(2),
            "base_rate": base_rate,
            "quoted_premium": quoted_premium,
            "converted": converted,
            "policy_year": policy_year,
        }
    )
    return df


# ── Helper functions ──────────────────────────────────────────────────────────

def _age_weights(ages: np.ndarray) -> np.ndarray:
    """Motorcycle rider age distribution — skewed young."""
    raw = np.exp(-0.5 * ((ages - 27) / 12) ** 2) + 0.3 * np.exp(-0.5 * ((ages - 50) / 15) ** 2)
    return raw / raw.sum()


def _age_effect(age: np.ndarray) -> np.ndarray:
    """U-shaped age effect: high risk for young/old riders."""
    return 0.6 * np.exp(-0.5 * ((age - 22) / 4) ** 2) + 0.3 * np.exp(-0.5 * ((age - 70) / 6) ** 2)


def _region_effect(region: np.ndarray) -> np.ndarray:
    effects = {
        "London": 0.30, "South East": 0.10, "South West": -0.05,
        "East of England": 0.00, "West Midlands": 0.15, "East Midlands": 0.05,
        "Yorkshire": 0.05, "North West": 0.10, "North East": 0.08,
        "Scotland": -0.05, "Wales": -0.10,
    }
    return np.array([effects[r] for r in region])


def _region_severity_effect(region: np.ndarray) -> np.ndarray:
    effects = {
        "London": 0.25, "South East": 0.10, "South West": -0.05,
        "East of England": 0.00, "West Midlands": 0.10, "East Midlands": 0.00,
        "Yorkshire": 0.00, "North West": 0.05, "North East": 0.00,
        "Scotland": -0.10, "Wales": -0.15,
    }
    return np.array([effects[r] for r in region])


def _occupation_effect(occupation: np.ndarray) -> np.ndarray:
    effects = {
        "Professional": -0.05, "Manual": 0.10, "Student": 0.20,
        "Retired": -0.10, "Self-employed": 0.05, "Other": 0.00,
    }
    return np.array([effects[o] for o in occupation])


def _conversion_curve(
    price_ratio: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sigmoid conversion probability as a function of price / technical rate."""
    base_conversion = 0.65
    elasticity = -3.0
    noise = rng.normal(0, 0.05, size=len(price_ratio))
    logit = elasticity * (price_ratio - 1.0) + noise
    return 1 / (1 + np.exp(-logit)) * base_conversion
