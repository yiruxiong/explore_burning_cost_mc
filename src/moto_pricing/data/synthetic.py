"""
Synthetic UK motorcycle insurance portfolio.

Generates a policy table and a separate claims table from a known data
generating process (DGP), so fitted models can be checked against ground
truth. Rating factor distributions and effect sizes are calibrated to match
well-documented patterns in the UK motorcycle market: frequency and severity
both rise sharply for young riders on high-powered sports bikes, no-claims
discount is strongly protective, and theft risk depends on security devices
and where the bike is kept overnight.

The frequency model (claims per policy per year) is Poisson:

    log(lambda) = log(exposure) + beta_0
                + beta_engine_cc * engine_cc_scaled
                + f(rider_age) + beta_licence_years * licence_years
                + beta_ncd * ncd_years + beta_bike_type[bike_type]
                + beta_area[area] + beta_convictions * has_convictions
                + beta_pillion * carries_pillion + beta_training * advanced_training
                + beta_mileage * log(annual_mileage / 5000)

Severity (incurred cost per claim) is Gamma, drawn per claim and specific to
the claim's peril (accident / theft / fire / third-party injury):

    log(mu) = gamma_0[peril] + gamma_engine_cc * engine_cc_scaled
            + gamma_bike_value * log(bike_value / 5000)
            + gamma_young * young_rider_flag

Security devices do not change claim severity directly; they change the
*mix* of perils a policy is exposed to (a tracked bike is far less likely to
generate a theft claim), which is the mechanism that matters for pricing.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Final

import numpy as np
import pandas as pd

BIKE_TYPES: Final[list[str]] = [
    "Scooter", "Naked", "Tourer", "Adventure", "Cruiser", "Sports", "Supersport",
]
# Approximate UK new-business mix by bike type.
BIKE_TYPE_PROBS: Final[list[float]] = [0.16, 0.22, 0.14, 0.13, 0.10, 0.17, 0.08]

# Typical engine size band (cc) sampled per bike type, before noise.
BIKE_TYPE_CC_RANGE: Final[dict[str, tuple[int, int]]] = {
    "Scooter": (50, 300),
    "Naked": (300, 1000),
    "Tourer": (600, 1800),
    "Adventure": (650, 1300),
    "Cruiser": (500, 1900),
    "Sports": (500, 1000),
    "Supersport": (750, 1300),
}

AREA_BANDS: Final[list[str]] = ["A", "B", "C", "D", "E", "F"]
AREA_PROBS: Final[list[float]] = [0.14, 0.19, 0.24, 0.21, 0.13, 0.09]

SECURITY_LEVELS: Final[list[str]] = ["none", "alarm", "tracker", "alarm_and_tracker"]
SECURITY_PROBS: Final[list[float]] = [0.45, 0.25, 0.10, 0.20]
# Multiplier applied to theft peril *share* — trackers are far more effective
# than alarms at preventing a stolen bike becoming a claim.
SECURITY_THEFT_MULTIPLIER: Final[dict[str, float]] = {
    "none": 1.00, "alarm": 0.70, "tracker": 0.35, "alarm_and_tracker": 0.22,
}

OVERNIGHT_PARKING: Final[list[str]] = ["garage", "driveway", "street"]
OVERNIGHT_PARKING_PROBS: Final[list[float]] = [0.45, 0.30, 0.25]
PARKING_THEFT_MULTIPLIER: Final[dict[str, float]] = {
    "garage": 0.45, "driveway": 0.85, "street": 1.60,
}

PERILS: Final[list[str]] = ["accident", "theft", "fire", "third_party_injury"]
BASE_PERIL_PROBS: Final[dict[str, float]] = {
    "accident": 0.62, "theft": 0.23, "fire": 0.03, "third_party_injury": 0.12,
}

TRUE_FREQ_PARAMS: Final[dict[str, float]] = {
    "intercept": -2.75,          # baseline frequency ~9% p.a. for an average rider
    "engine_cc_per_100": 0.028,  # per 100cc of engine size
    "rider_age_young": 0.85,     # additional log-frequency for riders under 25
    "rider_age_novice_old": 0.20,  # elevated frequency for very senior riders (75+)
    "licence_years": -0.018,     # per year licensed
    "ncd_years": -0.11,          # per year of no-claims discount
    "has_convictions": 0.40,
    "pillion": 0.08,             # carries a passenger at least sometimes
    "advanced_training": -0.18,  # completed BikeSafe / IAM or similar
    "log_mileage": 0.22,         # per unit of log(annual_mileage / 5000)
    "bike_type": {
        "Scooter": -0.05, "Naked": 0.00, "Tourer": -0.15, "Adventure": -0.08,
        "Cruiser": -0.10, "Sports": 0.40, "Supersport": 0.65,
    },
    "area": {"A": 0.0, "B": 0.08, "C": 0.17, "D": 0.28, "E": 0.42, "F": 0.55},
}

TRUE_SEV_PARAMS: Final[dict[str, float]] = {
    "peril_intercept": {
        "accident": 7.35,          # ~£1,550 baseline
        "theft": 7.55,             # ~£1,900 baseline (whole bike, minus excess)
        "fire": 7.70,               # ~£2,210 baseline
        "third_party_injury": 8.35,  # ~£4,230 baseline — injury claims are large
    },
    "engine_cc_per_100": 0.020,
    "log_bike_value": 0.30,      # per unit of log(bike_value / 5000)
    "young_rider": 0.18,
}

GAMMA_SHAPE: Final[float] = 1.8  # severity dispersion; CV = shape^-0.5 ~= 0.75


def _rider_age_effect(ages: np.ndarray) -> np.ndarray:
    """Non-linear rider age effect on log-frequency, U-shaped with a young peak."""
    effect = np.zeros(len(ages))
    effect[ages < 25] = TRUE_FREQ_PARAMS["rider_age_young"]
    effect[ages >= 75] = TRUE_FREQ_PARAMS["rider_age_novice_old"]
    blend_mask = (ages >= 25) & (ages < 30)
    blend_factor = (30 - ages[blend_mask]) / 5.0
    effect[blend_mask] = TRUE_FREQ_PARAMS["rider_age_young"] * blend_factor
    return effect


def generate_policies(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic UK motorcycle policy table.

    One row per policy, spanning five underwriting years (2019-2023) with
    realistic inception spread and an 8% early-cancellation rate. No
    exposure, claim, or premium columns are included — those are derived
    downstream (see :mod:`moto_pricing.data.build_dataset`).

    Parameters
    ----------
    n : int
        Number of policies to generate.
    seed : int
        Random seed. Different seeds give different, equally valid books.

    Returns
    -------
    pd.DataFrame
        One row per policy with rider, bike, and cover characteristics plus
        ``inception_date`` / ``expiry_date``.
    """
    rng = np.random.default_rng(seed)

    base_date = date(2019, 1, 1)
    total_days = 5 * 365
    inception_days = rng.integers(0, total_days, size=n)
    inception_dates = [base_date + timedelta(days=int(d)) for d in inception_days]

    is_cancellation = rng.random(n) < 0.08
    cancel_fraction = rng.uniform(0.05, 0.90, n)

    expiry_dates = []
    for i, inc in enumerate(inception_dates):
        try:
            full_expiry = date(inc.year + 1, inc.month, inc.day)
        except ValueError:
            full_expiry = date(inc.year + 1, inc.month, inc.day - 1)
        if is_cancellation[i]:
            term_days = (full_expiry - inc).days
            actual_days = max(1, int(term_days * cancel_fraction[i]))
            expiry_dates.append(inc + timedelta(days=actual_days))
        else:
            expiry_dates.append(full_expiry)

    # Rider age: UK motorcycle books skew older than car books — bikes are
    # disproportionately a mature/enthusiast purchase, with a smaller young
    # commuter-scooter segment.
    rider_ages = np.concatenate([
        rng.integers(17, 25, size=int(n * 0.10)),
        rng.integers(25, 40, size=int(n * 0.28)),
        rng.integers(40, 60, size=int(n * 0.38)),
        rng.integers(60, 75, size=int(n * 0.19)),
        rng.integers(75, 86, size=int(n * 0.05)),
    ])
    rider_ages = np.concatenate([rider_ages, rng.integers(17, 86, size=n - len(rider_ages))])
    rng.shuffle(rider_ages)

    max_licence = np.clip(rider_ages - 17, 0, 45).astype(int)
    licence_years = np.array([rng.integers(0, max(1, m + 1)) for m in max_licence])

    ncd_max = np.clip(licence_years // 2, 0, 9).astype(int)
    ncd_years = np.array([rng.integers(0, max(1, m + 1)) for m in ncd_max])

    conviction_probs = np.where(rider_ages < 28, 0.14, 0.05)
    has_convictions = rng.random(n) < conviction_probs
    conviction_points = np.where(
        has_convictions, rng.choice([3, 3, 3, 6, 6, 9], size=n), 0
    )

    bike_type = rng.choice(BIKE_TYPES, size=n, p=BIKE_TYPE_PROBS)
    engine_cc = np.array([
        rng.integers(*BIKE_TYPE_CC_RANGE[bt]) for bt in bike_type
    ]).astype(float)
    engine_cc = np.clip(engine_cc + rng.normal(0, 40, n), 50, 2000)

    bike_age = np.clip(rng.gamma(shape=2.2, scale=3.2, size=n).astype(int), 0, 25)
    # New bike list price scales with engine size; depreciates with bike age.
    list_price = 2_500 + 9.5 * engine_cc + rng.normal(0, 800, n)
    bike_value = np.clip(list_price * (0.92 ** bike_age), 500, None)

    security = rng.choice(SECURITY_LEVELS, size=n, p=SECURITY_PROBS)
    overnight_parking = rng.choice(OVERNIGHT_PARKING, size=n, p=OVERNIGHT_PARKING_PROBS)

    annual_mileage = np.clip(
        rng.lognormal(mean=8.4, sigma=0.55, size=n).astype(int), 500, 20_000
    )

    area = rng.choice(AREA_BANDS, size=n, p=AREA_PROBS)
    occupation_class = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.20, 0.30, 0.28, 0.15, 0.07])

    carries_pillion = rng.random(n) < np.where(bike_type == "Scooter", 0.35, 0.20)
    advanced_training = rng.random(n) < np.where(rider_ages < 30, 0.10, 0.06)

    tpft_prob = np.where(rider_ages < 25, 0.30, 0.10)
    policy_type = np.where(rng.random(n) < tpft_prob, "TPFT", "Comp")

    df = pd.DataFrame({
        "inception_date": inception_dates,
        "expiry_date": expiry_dates,
        "rider_age": rider_ages,
        "licence_years": licence_years,
        "ncd_years": ncd_years,
        "conviction_points": conviction_points,
        "bike_type": bike_type,
        "engine_cc": engine_cc.round(0).astype(int),
        "bike_age": bike_age,
        "bike_value": bike_value.round(2),
        "security": security,
        "overnight_parking": overnight_parking,
        "annual_mileage": annual_mileage,
        "area": area,
        "occupation_class": occupation_class,
        "carries_pillion": carries_pillion,
        "advanced_training": advanced_training,
        "policy_type": policy_type,
    })

    df.insert(0, "policy_id", np.arange(1, n + 1))
    return df


def _policy_term_exposure(df: pd.DataFrame) -> np.ndarray:
    """Simplified single-cohort earned exposure, used only to drive the DGP.

    The package's own :func:`burning_cost.data.exposure.calculate_exposure`
    is used downstream in :mod:`moto_pricing.data.build_dataset` for the
    real, calendar-accurate (and accident-year-split) exposure the models
    are trained on.
    """
    days = (
        pd.to_datetime(df["expiry_date"]) - pd.to_datetime(df["inception_date"])
    ).dt.days
    return (days / 365.25).clip(lower=0.0).to_numpy()


def _sample_perils(
    n_claims: int,
    theft_multiplier: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorised per-claim peril draw, with theft share scaled by security/parking."""
    base = np.array([BASE_PERIL_PROBS[p] for p in PERILS])
    theft_idx = PERILS.index("theft")

    probs = np.tile(base, (n_claims, 1))
    probs[:, theft_idx] *= theft_multiplier
    probs /= probs.sum(axis=1, keepdims=True)

    cum = probs.cumsum(axis=1)
    draws = rng.random(n_claims)[:, None]
    peril_idx = (draws < cum).argmax(axis=1)
    return np.array(PERILS)[peril_idx]


def generate_claims(policies: pd.DataFrame, seed: int = 43) -> pd.DataFrame:
    """
    Generate a claims table from the true frequency/severity DGP.

    Parameters
    ----------
    policies : pd.DataFrame
        Output of :func:`generate_policies`.
    seed : int
        Random seed, independent of the one used for the policy table.

    Returns
    -------
    pd.DataFrame
        One row per claim: ``claim_id``, ``policy_id``, ``claim_date``,
        ``peril``, ``incurred``.
    """
    rng = np.random.default_rng(seed)
    n = len(policies)
    exposure = _policy_term_exposure(policies)

    engine_cc_scaled = policies["engine_cc"].to_numpy() / 100.0
    young = (policies["rider_age"].to_numpy() < 25).astype(float)

    log_lambda = (
        TRUE_FREQ_PARAMS["intercept"]
        + TRUE_FREQ_PARAMS["engine_cc_per_100"] * engine_cc_scaled
        + _rider_age_effect(policies["rider_age"].to_numpy())
        + TRUE_FREQ_PARAMS["licence_years"] * policies["licence_years"].to_numpy()
        + TRUE_FREQ_PARAMS["ncd_years"] * policies["ncd_years"].to_numpy()
        + TRUE_FREQ_PARAMS["has_convictions"] * (policies["conviction_points"].to_numpy() > 0)
        + TRUE_FREQ_PARAMS["pillion"] * policies["carries_pillion"].to_numpy()
        + TRUE_FREQ_PARAMS["advanced_training"] * policies["advanced_training"].to_numpy()
        + TRUE_FREQ_PARAMS["log_mileage"]
        * np.log(policies["annual_mileage"].to_numpy() / 5_000)
        + policies["bike_type"].map(TRUE_FREQ_PARAMS["bike_type"]).to_numpy()
        + policies["area"].map(TRUE_FREQ_PARAMS["area"]).to_numpy()
        + np.log(np.clip(exposure, 1e-6, None))
    )

    claim_count = rng.poisson(np.exp(log_lambda))
    n_claims_total = int(claim_count.sum())

    policy_idx = np.repeat(np.arange(n), claim_count)
    claim_policy_ids = policies["policy_id"].to_numpy()[policy_idx]

    theft_multiplier = (
        policies["security"].map(SECURITY_THEFT_MULTIPLIER).to_numpy()
        * policies["overnight_parking"].map(PARKING_THEFT_MULTIPLIER).to_numpy()
    )[policy_idx]
    perils = _sample_perils(n_claims_total, theft_multiplier, rng)

    log_mu = (
        np.array([TRUE_SEV_PARAMS["peril_intercept"][p] for p in perils])
        + TRUE_SEV_PARAMS["engine_cc_per_100"] * engine_cc_scaled[policy_idx]
        + TRUE_SEV_PARAMS["log_bike_value"]
        * np.log(policies["bike_value"].to_numpy()[policy_idx] / 5_000)
        + TRUE_SEV_PARAMS["young_rider"] * young[policy_idx]
    )
    gamma_mean = np.exp(log_mu)
    incurred = rng.gamma(shape=GAMMA_SHAPE, scale=gamma_mean / GAMMA_SHAPE)

    inception = pd.to_datetime(policies["inception_date"]).to_numpy()[policy_idx]
    expiry = pd.to_datetime(policies["expiry_date"]).to_numpy()[policy_idx]
    span_days = ((expiry - inception) / np.timedelta64(1, "D")).astype(int)
    span_days = np.clip(span_days, 1, None)
    claim_offset_days = rng.integers(0, span_days + 1)
    claim_dates = inception + claim_offset_days.astype("timedelta64[D]")

    claims = pd.DataFrame({
        "claim_id": np.arange(1, n_claims_total + 1),
        "policy_id": claim_policy_ids,
        "claim_date": pd.to_datetime(claim_dates).date,
        "peril": perils,
        "incurred": incurred.round(2),
    })
    return claims


def true_expected_pure_premium(policies: pd.DataFrame) -> np.ndarray:
    """
    The DGP's exact expected annual pure premium — no sampling noise.

    Used to check model recovery against ground truth, and to simulate a
    competitor market that prices close to true risk (see
    :mod:`moto_pricing.journey.quotes`). Not available to the pricing models
    themselves, which only ever see realised ``claim_count`` / ``incurred``.
    """
    engine_cc_scaled = policies["engine_cc"].to_numpy() / 100.0
    young = (policies["rider_age"].to_numpy() < 25).astype(float)

    log_lambda = (
        TRUE_FREQ_PARAMS["intercept"]
        + TRUE_FREQ_PARAMS["engine_cc_per_100"] * engine_cc_scaled
        + _rider_age_effect(policies["rider_age"].to_numpy())
        + TRUE_FREQ_PARAMS["licence_years"] * policies["licence_years"].to_numpy()
        + TRUE_FREQ_PARAMS["ncd_years"] * policies["ncd_years"].to_numpy()
        + TRUE_FREQ_PARAMS["has_convictions"] * (policies["conviction_points"].to_numpy() > 0)
        + TRUE_FREQ_PARAMS["pillion"] * policies["carries_pillion"].to_numpy()
        + TRUE_FREQ_PARAMS["advanced_training"] * policies["advanced_training"].to_numpy()
        + TRUE_FREQ_PARAMS["log_mileage"]
        * np.log(policies["annual_mileage"].to_numpy() / 5_000)
        + policies["bike_type"].map(TRUE_FREQ_PARAMS["bike_type"]).to_numpy()
        + policies["area"].map(TRUE_FREQ_PARAMS["area"]).to_numpy()
    )
    expected_frequency = np.exp(log_lambda)

    theft_multiplier = (
        policies["security"].map(SECURITY_THEFT_MULTIPLIER).to_numpy()
        * policies["overnight_parking"].map(PARKING_THEFT_MULTIPLIER).to_numpy()
    )
    base = np.array([BASE_PERIL_PROBS[p] for p in PERILS])
    theft_idx = PERILS.index("theft")
    peril_probs = np.tile(base, (len(policies), 1))
    peril_probs[:, theft_idx] *= theft_multiplier
    peril_probs /= peril_probs.sum(axis=1, keepdims=True)

    peril_intercepts = np.array([TRUE_SEV_PARAMS["peril_intercept"][p] for p in PERILS])
    common_log = (
        TRUE_SEV_PARAMS["engine_cc_per_100"] * engine_cc_scaled
        + TRUE_SEV_PARAMS["log_bike_value"] * np.log(policies["bike_value"].to_numpy() / 5_000)
        + TRUE_SEV_PARAMS["young_rider"] * young
    )
    expected_severity_by_peril = np.exp(peril_intercepts)[None, :]  # (1, n_perils)
    expected_severity = np.exp(common_log) * (peril_probs * expected_severity_by_peril).sum(axis=1)

    return expected_frequency * expected_severity
