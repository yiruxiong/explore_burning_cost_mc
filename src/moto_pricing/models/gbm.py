"""
CatBoost frequency/severity ensemble.

Frequency uses a Poisson objective with ``baseline=log(exposure)`` — an
offset, exactly as in the GLM, and *not* also passed as a sample weight
(double-counting exposure is the single most common mistake described in
the burning-cost SHAP-relativities course, module 4). Severity uses a
Tweedie objective with ``variance_power=1.99`` (CatBoost requires it strictly
below 2), the closest GBM equivalent of a Gamma GLM, weighted by claim count.

Two boosted variants are trained per target (different depth/learning-rate
trade-offs) so :mod:`moto_pricing.models.ensemble` has more than one tree
model to blend — this is the "ensemble" half of the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from moto_pricing.features import CATEGORICAL_FEATURES, FREQUENCY_FEATURES, SEVERITY_FEATURES

_FREQ_VARIANTS: list[dict] = [
    {"iterations": 600, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0},
    {"iterations": 600, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 6.0},
]
_SEV_VARIANTS: list[dict] = [
    {"iterations": 500, "learning_rate": 0.05, "depth": 5, "l2_leaf_reg": 5.0},
    {"iterations": 500, "learning_rate": 0.03, "depth": 3, "l2_leaf_reg": 8.0},
]


def _cat_features(features: list[str]) -> list[str]:
    return [f for f in features if f in CATEGORICAL_FEATURES]


class GBMPricingModel:
    """A small CatBoost ensemble (bagged over hyperparameter variants) per target."""

    name = "gbm"

    def __init__(self, random_seed: int = 42) -> None:
        self._random_seed = random_seed
        self._freq_models: list[CatBoostRegressor] = []
        self._sev_models: list[CatBoostRegressor] = []

    def fit(self, train: pd.DataFrame, eval_set: pd.DataFrame | None = None) -> "GBMPricingModel":
        freq_cat = _cat_features(FREQUENCY_FEATURES)
        X_freq = train[FREQUENCY_FEATURES]
        log_exposure = np.log(np.clip(train["exposure"].to_numpy(), 1e-6, None))
        freq_pool = Pool(X_freq, train["claim_count"], cat_features=freq_cat, baseline=log_exposure)

        freq_eval_pool = None
        if eval_set is not None:
            log_exposure_eval = np.log(np.clip(eval_set["exposure"].to_numpy(), 1e-6, None))
            freq_eval_pool = Pool(
                eval_set[FREQUENCY_FEATURES], eval_set["claim_count"],
                cat_features=freq_cat, baseline=log_exposure_eval,
            )

        for i, params in enumerate(_FREQ_VARIANTS):
            model = CatBoostRegressor(
                loss_function="Poisson", random_seed=self._random_seed + i,
                verbose=False, **params,
            )
            model.fit(freq_pool, eval_set=freq_eval_pool, use_best_model=eval_set is not None)
            self._freq_models.append(model)

        sev_cat = _cat_features(SEVERITY_FEATURES)
        sev_train = train[train["claim_count"] > 0]
        sev_pool = Pool(
            sev_train[SEVERITY_FEATURES], sev_train["avg_severity"],
            cat_features=sev_cat, weight=sev_train["claim_count"],
        )

        sev_eval_pool = None
        if eval_set is not None:
            sev_eval = eval_set[eval_set["claim_count"] > 0]
            if len(sev_eval) > 0:
                sev_eval_pool = Pool(
                    sev_eval[SEVERITY_FEATURES], sev_eval["avg_severity"],
                    cat_features=sev_cat, weight=sev_eval["claim_count"],
                )

        for i, params in enumerate(_SEV_VARIANTS):
            model = CatBoostRegressor(
                loss_function="Tweedie:variance_power=1.99", random_seed=self._random_seed + i,
                verbose=False, **params,
            )
            model.fit(sev_pool, eval_set=sev_eval_pool, use_best_model=sev_eval_pool is not None)
            self._sev_models.append(model)

        return self

    def predict_frequency(self, df: pd.DataFrame) -> np.ndarray:
        """Annualised predicted claim frequency, averaged across the frequency variants."""
        exposure = df["exposure"].to_numpy(dtype=float)
        log_exposure = np.log(np.clip(exposure, 1e-6, None))
        pool = Pool(df[FREQUENCY_FEATURES], cat_features=_cat_features(FREQUENCY_FEATURES), baseline=log_exposure)
        preds = np.column_stack([m.predict(pool) for m in self._freq_models])
        return preds.mean(axis=1) / np.clip(exposure, 1e-6, None)

    def predict_severity(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted average cost per claim, averaged across the severity variants."""
        pool = Pool(df[SEVERITY_FEATURES], cat_features=_cat_features(SEVERITY_FEATURES))
        preds = np.column_stack([m.predict(pool) for m in self._sev_models])
        return preds.mean(axis=1)

    def predict_pure_premium(self, df: pd.DataFrame) -> np.ndarray:
        """Annualised risk premium: frequency x severity."""
        return self.predict_frequency(df) * self.predict_severity(df)

    @property
    def frequency_model(self) -> CatBoostRegressor:
        """The first (reference) frequency variant — used for SHAP relativities."""
        return self._freq_models[0]

    @property
    def severity_model(self) -> CatBoostRegressor:
        """The first (reference) severity variant — used for SHAP relativities."""
        return self._sev_models[0]
