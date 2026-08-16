"""
Quote generation: price a batch of prospects and simulate what the rest of
the market would have quoted them, so a price-competitiveness gap exists for
the conversion model to react to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from moto_pricing.data.synthetic import true_expected_pure_premium
from moto_pricing.pricing import PricingAssumptions, technical_to_gross


@dataclass(frozen=True)
class MarketAssumptions:
    # The rest of the panel prices to a similar risk view but isn't perfect.
    # Calibrated so the average quote is roughly market-competitive; expense
    # and profit loadings (see PricingAssumptions) are what actually push an
    # individual quote away from the market, not a built-in bias here.
    market_loading_factor: float = 2.0
    market_noise_sigma: float = 0.18


def simulate_quotes(
    prospects: pd.DataFrame,
    pricing_model,
    pricing_assumptions: PricingAssumptions | None = None,
    market_assumptions: MarketAssumptions | None = None,
    seed: int = 100,
) -> pd.DataFrame:
    """
    Price a batch of new-business prospects and simulate a competing market price.

    Parameters
    ----------
    prospects : pd.DataFrame
        Rating-factor rows for prospective policyholders (no claims history —
        this is a quote, not yet a policy). Must include ``exposure`` (use
        ``1.0`` for a full-term new-business quote).
    pricing_model
        A fitted model exposing ``predict_pure_premium(df)``, e.g.
        :class:`moto_pricing.models.EnsemblePricingModel`.
    pricing_assumptions, market_assumptions
        Loading assumptions for our price and the simulated competitor price.
    seed : int
        Random seed for market-price noise.

    Returns
    -------
    pd.DataFrame
        ``prospects`` plus ``risk_premium``, ``payable_premium`` (our quote),
        ``market_price`` and ``price_gap`` (our quote relative to the market,
        0 = matched, positive = we are more expensive).
    """
    rng = np.random.default_rng(seed)
    ma = market_assumptions or MarketAssumptions()

    risk_premium = pricing_model.predict_pure_premium(prospects)
    priced = technical_to_gross(risk_premium, pricing_assumptions)

    true_risk = true_expected_pure_premium(prospects)
    market_noise = rng.lognormal(mean=0.0, sigma=ma.market_noise_sigma, size=len(prospects))
    market_price = true_risk * ma.market_loading_factor * market_noise

    quotes = prospects.copy()
    quotes["risk_premium"] = priced["risk_premium"]
    quotes["gross_premium"] = priced["gross_premium"]
    quotes["payable_premium"] = priced["payable_premium"]
    quotes["market_price"] = market_price
    quotes["price_gap"] = quotes["payable_premium"] / quotes["market_price"] - 1.0
    return quotes
