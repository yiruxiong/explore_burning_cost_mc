"""
00_data_generation.py
=====================
Generate a synthetic UK motorcycle insurance portfolio.

Produces three calendar years (2021-2023) of policy-level data for ~30,000
policy-years across 50 fictional postcodes grouped into 5 rating regions.

Rating factors follow known UK motorcycle market dynamics:
- Rider age: U-shaped frequency curve peaking at 18-25 and 60+
- Engine cc: log-linear increase in severity
- NCD: discount tiers 0/1/2/3/4+ with claim-suppression effect
- Vehicle age: older bikes have lower severity (lower market value)
- Region: London/South-East uplift

A Sarmanov-style negative dependence is baked into the DGP: riders with
higher claim counts tend to have lower average severity per claim (NCD
suppression of small claims).
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Rating factor tables
# ---------------------------------------------------------------------------

AGE_FREQ_MULT = {
    (17, 21): 2.40,
    (22, 25): 1.80,
    (26, 35): 1.20,
    (36, 50): 1.00,
    (51, 60): 1.10,
    (61, 75): 1.50,
}

NCD_FREQ_MULT = {0: 1.45, 1: 1.20, 2: 1.05, 3: 0.90, 4: 0.75}
NCD_SEV_MULT  = {0: 0.85, 1: 0.90, 2: 0.95, 3: 1.00, 4: 1.05}  # NCD suppression baked in

REGION_MULT = {
    "London":       1.40,
    "South East":   1.20,
    "Midlands":     1.05,
    "North":        0.95,
    "Scotland":     0.90,
}

REGION_POSTCODES = {
    "London":     [f"E{i}"  for i in range(1, 11)],
    "South East": [f"SE{i}" for i in range(1, 11)],
    "Midlands":   [f"B{i}"  for i in range(1, 11)],
    "North":      [f"M{i}"  for i in range(1, 11)],
    "Scotland":   [f"G{i}"  for i in range(1, 11)],
}

BASE_FREQ  = 0.072   # ~7.2% claim frequency before rating factors
BASE_SEV   = 2_800.0 # £ base average cost per claim


def _age_mult(age: int) -> float:
    for (lo, hi), m in AGE_FREQ_MULT.items():
        if lo <= age <= hi:
            return m
    return 1.30  # default for out-of-range


def _engine_freq_mult(cc: int) -> float:
    return 0.75 + 0.50 * np.log1p(cc / 500)


def _engine_sev_mult(cc: int) -> float:
    return 0.60 + 0.80 * np.log1p(cc / 250)


def _vehicle_age_sev_mult(veh_age: int) -> float:
    return max(0.40, 1.0 - 0.05 * veh_age)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_portfolio(
    n_policies: int = 10_000,
    calendar_year: int = 2021,
    seed_offset: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(42 + seed_offset)

    # --- Policyholder characteristics ---
    age       = rng.integers(17, 76, n_policies)
    ncd       = rng.choice([0, 1, 2, 3, 4], n_policies, p=[0.15, 0.15, 0.20, 0.25, 0.25])
    engine_cc = rng.choice([125, 250, 400, 600, 800, 1000, 1200], n_policies,
                           p=[0.15, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05])
    veh_age   = rng.integers(0, 20, n_policies)

    region_list = list(REGION_MULT.keys())
    region_probs = [0.30, 0.25, 0.20, 0.15, 0.10]
    region       = rng.choice(region_list, n_policies, p=region_probs)

    postcode = np.array([
        rng.choice(REGION_POSTCODES[r])
        for r in region
    ])

    # Partial exposure (some mid-term cancellations)
    exposure = rng.uniform(0.3, 1.0, n_policies)

    # --- Frequency ---
    mu_n = np.array([
        BASE_FREQ
        * _age_mult(a)
        * NCD_FREQ_MULT[n]
        * _engine_freq_mult(e)
        * REGION_MULT[r]
        for a, n, e, r in zip(age, ncd, engine_cc, region)
    ]) * exposure

    claim_count = rng.poisson(mu_n)

    # --- Severity (conditional on at least one claim, Sarmanov-style NCD suppression) ---
    mu_s = np.array([
        BASE_SEV
        * NCD_SEV_MULT[n]
        * _engine_sev_mult(e)
        * _vehicle_age_sev_mult(v)
        for n, e, v in zip(ncd, engine_cc, veh_age)
    ])

    # Per-policy average severity (0 if no claims)
    avg_severity = np.where(
        claim_count > 0,
        rng.gamma(shape=2.0, scale=mu_s / 2.0),
        0.0,
    )

    total_incurred = claim_count * avg_severity

    return pd.DataFrame({
        "policy_id":     [f"{calendar_year}-{i:06d}" for i in range(n_policies)],
        "calendar_year": calendar_year,
        "rider_age":     age,
        "ncd_years":     ncd,
        "engine_cc":     engine_cc,
        "vehicle_age":   veh_age,
        "region":        region,
        "postcode":      postcode,
        "exposure":      exposure,
        "mu_freq":       mu_n,
        "claim_count":   claim_count,
        "mu_sev":        mu_s,
        "avg_severity":  avg_severity,
        "total_incurred": total_incurred,
    })


def generate_full_portfolio() -> pd.DataFrame:
    frames = []
    for yr, offset in [(2021, 0), (2022, 100), (2023, 200)]:
        frames.append(generate_portfolio(n_policies=10_000, calendar_year=yr,
                                         seed_offset=offset))
    df = pd.concat(frames, ignore_index=True)
    return df


def generate_quote_stream(n_quotes: int = 5_000, seed: int = 999) -> pd.DataFrame:
    """Quote stream for quote-to-price simulation (unseen data)."""
    rng = np.random.default_rng(seed)
    region_list = list(REGION_MULT.keys())
    region = rng.choice(region_list, n_quotes, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    return pd.DataFrame({
        "quote_id":    [f"Q{i:06d}" for i in range(n_quotes)],
        "rider_age":   rng.integers(17, 76, n_quotes),
        "ncd_years":   rng.choice([0, 1, 2, 3, 4], n_quotes, p=[0.15, 0.15, 0.20, 0.25, 0.25]),
        "engine_cc":   rng.choice([125, 250, 400, 600, 800, 1000, 1200], n_quotes,
                                  p=[0.15, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05]),
        "vehicle_age": rng.integers(0, 20, n_quotes),
        "region":      region,
    })


if __name__ == "__main__":
    df = generate_full_portfolio()
    print(f"Portfolio: {len(df):,} rows")
    print(df.groupby("calendar_year")[["claim_count", "total_incurred", "exposure"]].sum())
    print(f"\nOverall frequency: {df['claim_count'].sum() / df['exposure'].sum():.3%}")
    print(f"Overall avg severity (claims only): "
          f"£{df.loc[df['claim_count']>0, 'avg_severity'].mean():,.0f}")
