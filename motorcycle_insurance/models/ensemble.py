"""
Ensemble model pipeline for motorcycle insurance pricing.

Combines three complementary models:

1. **GLM** (statsmodels Poisson + Gamma) — interpretable baseline with
   region and occupation categorical effects.
2. **Penalized GLM / GAM** (insurance-gam Elastic Net) — regularised smooth
   effects for continuous rating factors (age, mileage, engine size).
3. **Credibility-adjusted GLM** — Bühlmann-Straub and Poisson-Gamma
   posterior blending for region and occupation thin-data segments.

The ensemble blends the three predictions via a simple weighted average,
with weights optimised on the validation set by minimising Poisson deviance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .frequency_severity import (
    fit_frequency_glm,
    fit_severity_glm,
    predict_pure_premium,
)
from .gam_model import (
    fit_gam_frequency,
    fit_gam_severity,
    predict_gam_pure_premium,
)
from .credibility import (
    fit_region_credibility,
    fit_occupation_credibility,
    apply_credibility_adjustment,
)


class EnsemblePricingModel:
    """Full ensemble pricing model for motorcycle insurance.

    Attributes set after :meth:`fit`
    --------------------------------
    glm_freq_ : statsmodels GLM result
    glm_sev_  : statsmodels GLM result
    gam_freq_ : PenalizedGLMInference (frequency)
    gam_sev_  : PenalizedGLMInference (severity)
    region_cred_ : dict (Bühlmann-Straub credibility by region)
    occ_cred_    : dict (Poisson-Gamma credibility by occupation)
    weights_ : np.ndarray shape (3,)
        Ensemble blend weights for [GLM, GAM, cred-GLM], summing to 1.
    """

    def __init__(self) -> None:
        self.glm_freq_ = None
        self.glm_sev_ = None
        self.gam_freq_ = None
        self.gam_sev_ = None
        self.region_cred_ = None
        self.occ_cred_ = None
        self.weights_: np.ndarray = np.array([1 / 3, 1 / 3, 1 / 3])

    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame | None = None,
    ) -> "EnsemblePricingModel":
        """Fit all component models and blend weights.

        Parameters
        ----------
        train:
            Training portfolio data.
        val:
            Validation data for blend-weight optimisation.  If ``None``,
            equal weights (1/3, 1/3, 1/3) are used.

        Returns
        -------
        self
        """
        # --- GLM components --------------------------------------------------
        print("  Fitting GLM frequency model...")
        self.glm_freq_ = fit_frequency_glm(train)

        print("  Fitting GLM severity model...")
        self.glm_sev_ = fit_severity_glm(train)

        # --- GAM components --------------------------------------------------
        print("  Fitting penalized GLM (GAM) frequency model...")
        self.gam_freq_ = fit_gam_frequency(train)

        print("  Fitting penalized GLM (GAM) severity model...")
        self.gam_sev_ = fit_gam_severity(train)

        # --- Credibility components ------------------------------------------
        print("  Fitting region credibility (Bühlmann-Straub)...")
        glm_freq_preds = self.glm_freq_.predict(train)
        self.region_cred_ = fit_region_credibility(train, glm_freq_preds)

        print("  Fitting occupation credibility (Poisson-Gamma)...")
        self.occ_cred_ = fit_occupation_credibility(train, glm_freq_preds)

        # --- Blend weights ---------------------------------------------------
        if val is not None:
            print("  Optimising ensemble weights on validation set...")
            self.weights_ = self._optimise_weights(val)
            print(f"  Ensemble weights: {self.weights_.round(3)}")

        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Produce ensemble pure-premium predictions.

        Parameters
        ----------
        data:
            Portfolio data to score.

        Returns
        -------
        np.ndarray of predicted pure premiums (£).
        """
        glm_pp = predict_pure_premium(self.glm_freq_, self.glm_sev_, data)
        gam_pp = predict_gam_pure_premium(self.gam_freq_, self.gam_sev_, data)
        cred_pp = apply_credibility_adjustment(
            data, glm_pp, self.region_cred_, self.occ_cred_
        )

        w = self.weights_
        return w[0] * glm_pp + w[1] * gam_pp + w[2] * cred_pp

    def _optimise_weights(self, val: pd.DataFrame) -> np.ndarray:
        """Find weights minimising Poisson deviance on the validation set."""
        glm_pp = predict_pure_premium(self.glm_freq_, self.glm_sev_, val)
        gam_pp = predict_gam_pure_premium(self.gam_freq_, self.gam_sev_, val)
        cred_pp = apply_credibility_adjustment(
            val, glm_pp, self.region_cred_, self.occ_cred_
        )

        actual = val["total_loss"].values.clip(0)

        def _poisson_deviance(w_raw: np.ndarray) -> float:
            w = np.exp(w_raw) / np.exp(w_raw).sum()
            pred = (w[0] * glm_pp + w[1] * gam_pp + w[2] * cred_pp).clip(1e-6)
            return float(2 * np.sum(
                np.where(actual > 0, actual * np.log(actual / pred), 0)
                - (actual - pred)
            ))

        result = minimize(
            _poisson_deviance,
            x0=np.zeros(3),
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-4},
        )
        w_opt = np.exp(result.x) / np.exp(result.x).sum()
        return w_opt
