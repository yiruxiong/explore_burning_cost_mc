"""
Quote-to-pricing journey simulation.

Models the journey from a raw customer quote request through to a final
quoted premium.  Steps:

1. **Risk scoring** — run the ensemble model to obtain a pure technical premium.
2. **Loading** — apply expense ratio, profit margin, and reinsurance loading.
3. **Rating factor validation** — flag high-risk or out-of-appetite risks.
4. **Quote generation** — produce a final quoted premium with a validity window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class PricingConfig:
    """Pricing configuration parameters."""

    expense_ratio: float = 0.25
    """Expense loading as a proportion of the technical premium."""

    profit_margin: float = 0.05
    """Target profit margin as a proportion of premium."""

    reinsurance_loading: float = 0.03
    """Reinsurance cost loading as a proportion of premium."""

    minimum_premium: float = 150.0
    """Minimum quoted premium (£)."""

    maximum_premium: float = 5_000.0
    """Maximum quoted premium (£); above this the risk is referred."""

    high_risk_freq_threshold: float = 0.50
    """Expected claims-per-year above which a risk is flagged as high-risk."""

    quote_validity_days: int = 30
    """Number of days a quote is valid."""


@dataclass
class QuoteResult:
    """Result of a single quote journey."""

    policy_id: int
    technical_premium: float
    gross_premium: float
    quoted_premium: float
    is_referred: bool
    refer_reason: str
    loading_factor: float


def run_quote_to_pricing(
    quotes: pd.DataFrame,
    ensemble_model,
    config: PricingConfig | None = None,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Simulate the quote-to-pricing journey for a batch of quote requests.

    Parameters
    ----------
    quotes:
        DataFrame of quote requests.  Must contain all rating factors
        expected by the ensemble model.
    ensemble_model:
        Fitted :class:`~motorcycle_insurance.models.ensemble.EnsemblePricingModel`.
    config:
        :class:`PricingConfig` with loading parameters.  Uses defaults if
        ``None``.
    rng_seed:
        Random seed for any stochastic elements.

    Returns
    -------
    pd.DataFrame
        One row per quote with columns:
        ``policy_id``, ``technical_premium``, ``gross_premium``,
        ``quoted_premium``, ``is_referred``, ``refer_reason``,
        ``loading_factor``.
    """
    if config is None:
        config = PricingConfig()

    rng = np.random.default_rng(rng_seed)

    # Step 1: technical pure premium from ensemble model
    technical_premium = ensemble_model.predict(quotes)

    # Step 2: gross premium = technical / (1 - expenses - profit - RI)
    net_ratio = 1.0 - config.expense_ratio - config.profit_margin - config.reinsurance_loading
    net_ratio = max(net_ratio, 0.5)  # safety floor
    gross_premium = technical_premium / net_ratio

    # Step 3: competitive market noise (±5%)
    market_noise = rng.uniform(0.95, 1.05, size=len(quotes))
    quoted_premium = (gross_premium * market_noise).round(2)

    # Step 4: apply floor and ceiling
    quoted_premium = quoted_premium.clip(config.minimum_premium, None)

    # Step 5: referrals
    glm_expected_freq = ensemble_model.glm_freq_.predict(quotes)
    is_referred = (
        (quoted_premium > config.maximum_premium)
        | (glm_expected_freq > config.high_risk_freq_threshold)
    )
    refer_reason = np.where(
        quoted_premium > config.maximum_premium,
        "premium_too_high",
        np.where(
            glm_expected_freq > config.high_risk_freq_threshold,
            "high_risk_frequency",
            "",
        ),
    )

    # Cap referred quotes at maximum
    quoted_premium = quoted_premium.clip(None, config.maximum_premium)

    loading_factor = np.where(
        technical_premium > 0, quoted_premium / technical_premium, 1.0
    )

    result = pd.DataFrame(
        {
            "policy_id": quotes["policy_id"].values,
            "technical_premium": technical_premium.round(2),
            "gross_premium": gross_premium.round(2),
            "quoted_premium": quoted_premium,
            "is_referred": is_referred,
            "refer_reason": refer_reason,
            "loading_factor": loading_factor.round(3),
        }
    )
    return result
