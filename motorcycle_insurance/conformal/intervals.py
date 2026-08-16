"""
Conformal prediction intervals for motorcycle insurance pure premiums.

Uses :class:`~insurance_conformal.InsuranceConformalPredictor` to produce
distribution-free prediction intervals with guaranteed marginal coverage.

The calibration split follows the temporal ordering of the data:
  - Training years  : fit ensemble model
  - Calibration year: fit conformal calibration (non-conformity scores)
  - Scoring year    : produce intervals for new business

A simple model adapter wraps the ensemble to expose the ``predict(X)``
interface expected by :class:`~insurance_conformal.InsuranceConformalPredictor`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from insurance_conformal import InsuranceConformalPredictor


class EnsemblePredictor:
    """Thin sklearn-compatible wrapper around the ensemble model.

    :class:`~insurance_conformal.InsuranceConformalPredictor` calls
    ``model.predict(X)`` where X is passed from ``calibrate`` and
    ``predict_interval``.  This wrapper routes those calls back to
    :meth:`~motorcycle_insurance.models.ensemble.EnsemblePricingModel.predict`.
    """

    def __init__(self, ensemble_model) -> None:
        self._model = ensemble_model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


def build_conformal_predictor(
    ensemble_model,
    calibration_df: pd.DataFrame,
    alpha: float = 0.10,
) -> InsuranceConformalPredictor:
    """Build a calibrated conformal predictor from the ensemble model.

    Parameters
    ----------
    ensemble_model:
        Fitted :class:`~motorcycle_insurance.models.ensemble.EnsemblePricingModel`.
    calibration_df:
        Held-out calibration data (independent year / holdout set).
        Must contain ``total_loss`` and ``exposure`` columns.
    alpha:
        Miscoverage rate.  ``alpha=0.10`` → 90% prediction intervals.

    Returns
    -------
    Calibrated :class:`~insurance_conformal.InsuranceConformalPredictor`.
    """
    wrapper = EnsemblePredictor(ensemble_model)

    cp = InsuranceConformalPredictor(
        model=wrapper,
        nonconformity="pearson_weighted",
        distribution="tweedie",
        tweedie_power=1.5,
    )

    y_cal = calibration_df["total_loss"].values
    exposure_cal = calibration_df["exposure"].values

    cp.calibrate(X_cal=calibration_df, y_cal=y_cal, exposure=exposure_cal)
    return cp


def predict_premium_intervals(
    cp: InsuranceConformalPredictor,
    scoring_df: pd.DataFrame,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Produce prediction intervals for a scoring portfolio.

    Parameters
    ----------
    cp:
        Calibrated conformal predictor from :func:`build_conformal_predictor`.
    scoring_df:
        Portfolio data to score.
    alpha:
        Miscoverage rate.

    Returns
    -------
    pd.DataFrame with columns ``lower``, ``point``, ``upper`` (£).
    """
    intervals: pl.DataFrame = cp.predict_interval(X_test=scoring_df, alpha=alpha)
    return intervals.to_pandas()
