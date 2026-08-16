"""
GLM baseline: Poisson frequency, Gamma severity, both log-link.

This is the pricing actuary's traditional workhorse and the benchmark every
GBM has to beat. Frequency uses ``log(exposure)`` as a fixed offset (not a
weight — see the module docstring in ``burning_cost.shap_relativities._core``
for why that distinction matters). Severity is fit on claim rows only,
weighted by claim count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from moto_pricing.features import FREQUENCY_FEATURES, SEVERITY_FEATURES
from moto_pricing.models.design_matrix import DesignMatrixBuilder, log_offset


class GLMPricingModel:
    """Two-part Poisson/Gamma GLM producing frequency, severity and pure premium."""

    name = "glm"

    def __init__(self) -> None:
        self._freq_design = DesignMatrixBuilder(features=FREQUENCY_FEATURES)
        self._sev_design = DesignMatrixBuilder(features=SEVERITY_FEATURES)
        self._freq_result = None
        self._sev_result = None

    def fit(self, train: pd.DataFrame) -> "GLMPricingModel":
        X_freq = self._freq_design.fit(train).transform(train)
        self._freq_result = sm.GLM(
            train["claim_count"],
            X_freq,
            family=sm.families.Poisson(link=sm.families.links.Log()),
            offset=log_offset(train["exposure"]),
        ).fit()

        sev_train = train[train["claim_count"] > 0]
        X_sev = self._sev_design.fit(sev_train).transform(sev_train)
        self._sev_result = sm.GLM(
            sev_train["avg_severity"],
            X_sev,
            family=sm.families.Gamma(link=sm.families.links.Log()),
            freq_weights=sev_train["claim_count"].to_numpy(),
        ).fit()
        return self

    def predict_frequency(self, df: pd.DataFrame) -> np.ndarray:
        """Annualised predicted claim frequency (claims per exposure-year)."""
        X = self._freq_design.transform(df)
        exposure = df["exposure"].to_numpy(dtype=float)
        pred_count = self._freq_result.predict(X, offset=log_offset(df["exposure"]))
        return pred_count.to_numpy() / np.clip(exposure, 1e-6, None)

    def predict_severity(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted average cost per claim."""
        X = self._sev_design.transform(df)
        return self._sev_result.predict(X).to_numpy()

    def predict_pure_premium(self, df: pd.DataFrame) -> np.ndarray:
        """Annualised risk premium: frequency x severity."""
        return self.predict_frequency(df) * self.predict_severity(df)
