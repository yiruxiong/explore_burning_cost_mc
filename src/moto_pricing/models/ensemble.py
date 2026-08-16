"""
Ensemble pricing model: a convex blend of the GLM and the CatBoost ensemble.

Blend weights are chosen on a held-out slice of the training years (not the
final out-of-time test year), by grid search over the weight on the GBM,
minimising exposure/claim-weighted squared error against actuals. This is a
simple, transparent stand-in for proper stacking that is easy to audit — a
pricing committee can see exactly what weight each base model got and why.
Once the weight is chosen, both base models are refit on the full training
window so nothing is left on the table for the deployed model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from moto_pricing.models.gbm import GBMPricingModel
from moto_pricing.models.glm import GLMPricingModel

_WEIGHT_GRID = np.linspace(0.0, 1.0, 21)


def _best_blend_weight(actual: np.ndarray, glm_pred: np.ndarray, gbm_pred: np.ndarray, weight: np.ndarray) -> float:
    errors = [
        np.average((weight_gbm * gbm_pred + (1 - weight_gbm) * glm_pred - actual) ** 2, weights=weight)
        for weight_gbm in _WEIGHT_GRID
    ]
    return float(_WEIGHT_GRID[int(np.argmin(errors))])


class EnsemblePricingModel:
    """Blends :class:`GLMPricingModel` and :class:`GBMPricingModel`."""

    name = "ensemble"

    def __init__(self, random_seed: int = 42) -> None:
        self._random_seed = random_seed
        self.glm = GLMPricingModel()
        self.gbm = GBMPricingModel(random_seed=random_seed)
        self.freq_weight_: float | None = None
        self.sev_weight_: float | None = None

    def fit(
        self,
        train: pd.DataFrame,
        stacking_val_year: int,
        fitted_glm: GLMPricingModel | None = None,
        fitted_gbm: GBMPricingModel | None = None,
    ) -> "EnsemblePricingModel":
        """
        Fit the ensemble.

        Parameters
        ----------
        train : pd.DataFrame
            The full training window (see :func:`moto_pricing.features.out_of_time_split`).
        stacking_val_year : int
            Accident year within ``train`` reserved for choosing blend weights.
            Earlier years are used to fit the base learners for weight search.
        fitted_glm, fitted_gbm : optional
            Base learners already fit on the *full* ``train`` window (e.g. the
            standalone models a pipeline also reports on their own). Reused as
            the ensemble's deployed base learners instead of refitting, since
            fitting a GBM on the full book twice is pure waste. Blend weights
            are still learned from scratch on a held-out slice of ``train``
            regardless, to avoid leaking the stacking validation year into
            the base learners used to choose the weights.
        """
        base_fit = train[train["accident_year"] < stacking_val_year]
        stack_val = train[train["accident_year"] == stacking_val_year]

        search_glm = GLMPricingModel().fit(base_fit)
        search_gbm = GBMPricingModel(random_seed=0).fit(base_fit, eval_set=stack_val)

        actual_freq = (stack_val["claim_count"] / stack_val["exposure"]).to_numpy()
        self.freq_weight_ = _best_blend_weight(
            actual_freq,
            search_glm.predict_frequency(stack_val),
            search_gbm.predict_frequency(stack_val),
            stack_val["exposure"].to_numpy(),
        )

        sev_val = stack_val[stack_val["claim_count"] > 0]
        self.sev_weight_ = _best_blend_weight(
            sev_val["avg_severity"].to_numpy(),
            search_glm.predict_severity(sev_val),
            search_gbm.predict_severity(sev_val),
            sev_val["claim_count"].to_numpy(),
        )

        self.glm = fitted_glm or GLMPricingModel().fit(train)
        self.gbm = fitted_gbm or GBMPricingModel(random_seed=self._random_seed).fit(train, eval_set=stack_val)
        return self

    def predict_frequency(self, df: pd.DataFrame) -> np.ndarray:
        w = self.freq_weight_
        return w * self.gbm.predict_frequency(df) + (1 - w) * self.glm.predict_frequency(df)

    def predict_severity(self, df: pd.DataFrame) -> np.ndarray:
        w = self.sev_weight_
        return w * self.gbm.predict_severity(df) + (1 - w) * self.glm.predict_severity(df)

    def predict_pure_premium(self, df: pd.DataFrame) -> np.ndarray:
        """Annualised risk premium: frequency x severity."""
        return self.predict_frequency(df) * self.predict_severity(df)
