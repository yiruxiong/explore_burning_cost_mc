"""
Pricing-to-sell journey simulation.

Models the journey from a quoted premium through to a policy sale or
lapse.  Captures price elasticity, competitive market dynamics, and
cross-sell / renewal behaviour in the UK motorcycle personal lines market.

Key concepts
------------
- **Conversion probability** — probability a quoted customer buys, modelled
  as a sigmoid function of the price ratio (quote / market average).
- **Market price index** — competitive rate for each risk segment.
- **Renewal rate** — probability an existing customer renews at the new
  quoted premium.
- **Loss ratio at point of sale** — estimated profitability of converted
  business.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MarketConfig:
    """Market and elasticity parameters."""

    base_conversion_rate: float = 0.60
    """Expected conversion when quoting exactly at technical rate."""

    price_elasticity: float = -2.5
    """Logit slope on log(quote/market_average); more negative = more elastic."""

    renewal_base_rate: float = 0.75
    """Base renewal probability at flat rate."""

    renewal_elasticity: float = -3.0
    """Logit slope for renewals; customers are more rate-sensitive on renewal."""

    market_noise_std: float = 0.10
    """Standard deviation of log-normal market price noise per risk."""


def simulate_market_price(
    technical_premium: np.ndarray,
    config: MarketConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a competitive market reference price for each risk.

    The market price is modelled as the technical premium multiplied by a
    log-normal noise term, representing the spread of prices in the market.

    Parameters
    ----------
    technical_premium:
        Array of technical pure premiums.
    config:
        Market configuration parameters.
    rng:
        NumPy random generator.

    Returns
    -------
    np.ndarray of market reference prices.
    """
    noise = np.exp(rng.normal(0, config.market_noise_std, size=len(technical_premium)))
    return technical_premium * noise


def compute_conversion_probability(
    quoted_premium: np.ndarray,
    market_price: np.ndarray,
    config: MarketConfig,
) -> np.ndarray:
    """Compute probability of policy sale given quoted vs. market price.

    Uses a logistic function of log(quote/market), so the conversion
    decreases smoothly as the quote increases above market.

    Parameters
    ----------
    quoted_premium:
        Array of quoted premiums (£).
    market_price:
        Array of simulated market reference prices (£).
    config:
        Market parameters including ``base_conversion_rate`` and
        ``price_elasticity``.

    Returns
    -------
    np.ndarray of conversion probabilities in [0, 1].
    """
    log_price_ratio = np.log(quoted_premium / market_price.clip(1e-6))
    logit = config.price_elasticity * log_price_ratio
    raw_prob = 1.0 / (1.0 + np.exp(-logit))
    return raw_prob * config.base_conversion_rate


def compute_renewal_probability(
    quoted_premium: np.ndarray,
    previous_premium: np.ndarray,
    config: MarketConfig,
) -> np.ndarray:
    """Compute renewal probability given premium change at renewal.

    Customers are more likely to shop around when their premium increases
    sharply.

    Parameters
    ----------
    quoted_premium:
        Renewal quoted premiums (£).
    previous_premium:
        Previous year's premiums (£).
    config:
        Market parameters.

    Returns
    -------
    np.ndarray of renewal probabilities.
    """
    rate_change = (quoted_premium - previous_premium) / previous_premium.clip(1e-6)
    logit = config.renewal_elasticity * rate_change
    raw_prob = 1.0 / (1.0 + np.exp(-logit))
    return raw_prob * config.renewal_base_rate


def run_price_to_sell(
    quotes_df: pd.DataFrame,
    pricing_result: pd.DataFrame,
    config: MarketConfig | None = None,
    renewal: bool = False,
    previous_premium_col: str = "previous_premium",
    rng_seed: int = 99,
) -> pd.DataFrame:
    """Simulate the price-to-sell journey for a batch of quoted risks.

    Parameters
    ----------
    quotes_df:
        Quote request data (rating factors).
    pricing_result:
        Output of :func:`~pipelines.quote_to_price.run_quote_to_pricing`.
    config:
        :class:`MarketConfig` parameters.  Uses defaults if ``None``.
    renewal:
        If ``True``, use renewal conversion model.  Requires
        ``previous_premium_col`` in ``quotes_df``.
    previous_premium_col:
        Column in ``quotes_df`` with previous year's premium (renewals only).
    rng_seed:
        Random seed.

    Returns
    -------
    pd.DataFrame
        Original pricing result augmented with columns:
        ``market_price``, ``conversion_prob``, ``converted``,
        ``written_premium``, ``loss_ratio_estimate``.
    """
    if config is None:
        config = MarketConfig()

    rng = np.random.default_rng(rng_seed)

    quoted = pricing_result["quoted_premium"].values
    technical = pricing_result["technical_premium"].values.clip(1e-6)

    market_price = simulate_market_price(technical, config, rng)

    if renewal and previous_premium_col in quotes_df.columns:
        prev = quotes_df[previous_premium_col].values.clip(1e-6)
        conversion_prob = compute_renewal_probability(quoted, prev, config)
    else:
        conversion_prob = compute_conversion_probability(quoted, market_price, config)

    # Referred risks have lower conversion (manual process)
    is_referred = pricing_result["is_referred"].values
    conversion_prob = np.where(is_referred, conversion_prob * 0.5, conversion_prob)

    converted = rng.binomial(1, conversion_prob.clip(0, 1))
    written_premium = converted * quoted

    # Estimated loss ratio = technical / quoted (inverted loading)
    loss_ratio_estimate = np.where(
        quoted > 0, technical / quoted, 1.0
    ).round(3)

    result = pricing_result.copy()
    result["market_price"] = market_price.round(2)
    result["conversion_prob"] = conversion_prob.round(4)
    result["converted"] = converted
    result["written_premium"] = written_premium.round(2)
    result["loss_ratio_estimate"] = loss_ratio_estimate

    return result


def summarise_portfolio_economics(sell_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise key portfolio economics from the price-to-sell output.

    Parameters
    ----------
    sell_df:
        Output of :func:`run_price_to_sell`.

    Returns
    -------
    pd.DataFrame with one row per key metric:
    ``metric``, ``value``.
    """
    sold = sell_df[sell_df["converted"] == 1]

    metrics = {
        "total_quotes": len(sell_df),
        "total_converted": int(sell_df["converted"].sum()),
        "overall_conversion_rate": round(sell_df["converted"].mean(), 4),
        "referred_quotes": int(sell_df["is_referred"].sum()),
        "referred_conversion_rate": round(
            sell_df.loc[sell_df["is_referred"], "converted"].mean(), 4
        ) if sell_df["is_referred"].any() else 0.0,
        "total_written_premium": round(sell_df["written_premium"].sum(), 2),
        "average_written_premium": round(sold["written_premium"].mean(), 2) if len(sold) else 0.0,
        "average_technical_premium": round(sold["technical_premium"].mean(), 2) if len(sold) else 0.0,
        "average_loading_factor": round(sold["loading_factor"].mean(), 3) if len(sold) else 0.0,
        "weighted_loss_ratio": round(
            (sold["loss_ratio_estimate"] * sold["written_premium"]).sum()
            / sold["written_premium"].sum(), 4
        ) if len(sold) and sold["written_premium"].sum() > 0 else 0.0,
    }

    return pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    )
