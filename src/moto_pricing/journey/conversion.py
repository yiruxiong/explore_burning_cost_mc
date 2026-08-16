"""New-business conversion: price-elasticity model turning a quote into a bind decision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConversionAssumptions:
    base_conversion_logit: float = -0.60  # ~35% baseline conversion when priced at market
    price_elasticity: float = 3.2         # logit points lost per 100% price premium over market
    idiosyncratic_noise_sd: float = 0.35  # brand/add-on/service factors unrelated to price


def simulate_new_business_conversion(
    quotes: pd.DataFrame, assumptions: ConversionAssumptions | None = None, seed: int = 101
) -> pd.DataFrame:
    """
    Simulate whether each quoted prospect binds the policy.

    A logistic price-elasticity model: conversion probability falls as our
    quote gets more expensive than the simulated market price, plus
    idiosyncratic noise standing in for everything price can't explain
    (service reputation, add-ons, plain inertia).

    Parameters
    ----------
    quotes : pd.DataFrame
        Output of :func:`moto_pricing.journey.quotes.simulate_quotes`. Must
        contain ``price_gap``.
    assumptions : ConversionAssumptions | None
    seed : int

    Returns
    -------
    pd.DataFrame
        ``quotes`` plus ``conversion_probability`` and ``bound`` (bool).
    """
    a = assumptions or ConversionAssumptions()
    rng = np.random.default_rng(seed)

    logit = (
        a.base_conversion_logit
        - a.price_elasticity * quotes["price_gap"].to_numpy()
        + rng.normal(0, a.idiosyncratic_noise_sd, len(quotes))
    )
    conversion_probability = 1 / (1 + np.exp(-logit))

    result = quotes.copy()
    result["conversion_probability"] = conversion_probability
    result["bound"] = rng.random(len(quotes)) < conversion_probability
    return result
