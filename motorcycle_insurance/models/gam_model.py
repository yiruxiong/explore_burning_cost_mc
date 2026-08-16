"""
Penalized GLM model for motorcycle insurance — the GAM-style component.

Uses :class:`~insurance_gam.penalized_glm_inference.PenalizedGLMInference`
to fit Elastic Net penalized Poisson (frequency) and Gamma (severity) GLMs.
Penalisation smooths the continuous rating-factor effects analogously to
spline-based GAMs, yielding regularised coefficient estimates and
bias-corrected confidence intervals for each rating factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from insurance_gam.penalized_glm_inference import PenalizedGLMInference

# Continuous features — log-transformed before fitting
FREQ_FEATURES = [
    "log_age", "log_years_licensed", "log_bike_cc",
    "log_bike_age", "log_mileage", "ncb",
]
SEV_FEATURES = [
    "log_bike_value", "log_bike_cc",
]


def _build_freq_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build numeric design matrix for frequency model."""
    X = pd.DataFrame(
        {
            "log_age": np.log(df["age"].clip(17, None)),
            "log_years_licensed": np.log1p(df["years_licensed"]),
            "log_bike_cc": np.log(df["bike_cc"].clip(50, None)),
            "log_bike_age": np.log1p(df["bike_age"]),
            "log_mileage": np.log(df["annual_mileage"].clip(100, None)),
            "ncb": df["ncb"].astype(float),
        }
    )
    return X


def _build_sev_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build numeric design matrix for severity model."""
    X = pd.DataFrame(
        {
            "log_bike_value": np.log(df["bike_value"].clip(100, None)),
            "log_bike_cc": np.log(df["bike_cc"].clip(50, None)),
        }
    )
    return X


def fit_gam_frequency(train: pd.DataFrame) -> PenalizedGLMInference:
    """Fit a penalized Poisson frequency model.

    Parameters
    ----------
    train:
        Training portfolio data.

    Returns
    -------
    Fitted :class:`~insurance_gam.penalized_glm_inference.PenalizedGLMInference`.
    """
    X = _build_freq_matrix(train)
    y = train["claim_count"].astype(float)
    exposure = train["exposure"].values

    model = PenalizedGLMInference(
        family="poisson",
        alpha=0.5,
        l1_ratio=0.5,
        random_state=42,
    )
    model.fit(X, y, exposure=exposure)
    return model


def fit_gam_severity(train: pd.DataFrame) -> PenalizedGLMInference:
    """Fit a penalized Gamma severity model (claims-only rows).

    Parameters
    ----------
    train:
        Training portfolio data.

    Returns
    -------
    Fitted :class:`~insurance_gam.penalized_glm_inference.PenalizedGLMInference`.
    """
    claims = train[train["claim_count"] > 0].copy()
    X = _build_sev_matrix(claims)
    y = claims["avg_severity"].astype(float)

    model = PenalizedGLMInference(
        family="gamma",
        alpha=0.5,
        l1_ratio=0.5,
        random_state=42,
    )
    model.fit(X, y)
    return model


def predict_gam_frequency(
    model: PenalizedGLMInference,
    data: pd.DataFrame,
) -> np.ndarray:
    """Score data through the penalized frequency model.

    Returns expected claim counts (per-policy, exposure-adjusted).

    Note: ``PenalizedGLMInference`` does not expose a ``predict`` method.
    We reconstruct predictions from the fitted coefficients and the internal
    scaler.  This is intentional — the library is designed for inference
    (CIs), not prediction pipelines.
    """
    X = _build_freq_matrix(data)
    X_scaled = model._scaler.transform(X.values)
    log_mu = X_scaled @ model.coef_penalized_ + model.intercept_
    return np.exp(log_mu.clip(-15, 15)) * data["exposure"].values


def predict_gam_severity(
    model: PenalizedGLMInference,
    data: pd.DataFrame,
) -> np.ndarray:
    """Score data through the penalized severity model.

    Returns expected average severity per claim.

    Note: ``PenalizedGLMInference`` does not expose a ``predict`` method.
    We reconstruct predictions from the fitted coefficients and the internal
    scaler — see :func:`predict_gam_frequency` for the rationale.
    """
    X = _build_sev_matrix(data)
    X_scaled = model._scaler.transform(X.values)
    log_mu = X_scaled @ model.coef_penalized_ + model.intercept_
    return np.exp(log_mu.clip(-15, 15))


def predict_gam_pure_premium(
    freq_model: PenalizedGLMInference,
    sev_model: PenalizedGLMInference,
    data: pd.DataFrame,
) -> np.ndarray:
    """Combined pure-premium prediction from GAM frequency × severity."""
    freq = predict_gam_frequency(freq_model, data)
    sev = predict_gam_severity(sev_model, data)
    return freq * sev
