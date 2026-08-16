"""Feature lists and the out-of-time train/test split shared by every model."""

from __future__ import annotations

import pandas as pd

CATEGORICAL_FEATURES: list[str] = [
    "bike_type", "security", "overnight_parking", "area", "policy_type",
]

FREQUENCY_FEATURES: list[str] = CATEGORICAL_FEATURES + [
    "rider_age", "licence_years", "ncd_years", "conviction_points",
    "engine_cc", "bike_age", "annual_mileage", "carries_pillion",
    "advanced_training", "occupation_class",
]

SEVERITY_FEATURES: list[str] = CATEGORICAL_FEATURES + [
    "rider_age", "engine_cc", "bike_value", "bike_age",
]

LAST_TRAIN_YEAR = 2022  # 2023 is held out as the out-of-time test year.


def out_of_time_split(modelling: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the modelling frame by accident year, not randomly.

    A random split leaks information: a GBM can otherwise exploit the fact
    that a policy's neighbours (same rider, similar mid-term factors) sit in
    both train and test. Splitting on time is what a real pricing team does
    to estimate performance on business it hasn't seen yet.

    Returns
    -------
    (train, test) : tuple[pd.DataFrame, pd.DataFrame]
        ``train`` is accident years up to and including ``LAST_TRAIN_YEAR``;
        ``test`` is everything after.
    """
    train = modelling[modelling["accident_year"] <= LAST_TRAIN_YEAR].reset_index(drop=True)
    test = modelling[modelling["accident_year"] > LAST_TRAIN_YEAR].reset_index(drop=True)
    return train, test
