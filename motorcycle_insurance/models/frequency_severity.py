"""
Frequency / severity GLM models for motorcycle insurance pricing.

Uses statsmodels Poisson GLM for claim frequency and Gamma GLM for
claim severity.  Provides a combined pure-premium prediction and a
ConditionalFreqSev model that captures frequency-severity dependence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from insurance_frequency_severity import ConditionalFreqSev


FREQ_FORMULA = (
    "claim_count ~ "
    "np.log(age) + np.log1p(years_licensed) + np.log(bike_cc) "
    "+ np.log1p(bike_age) + np.log(annual_mileage) + ncb "
    "+ C(region) + C(occupation)"
)

SEV_FORMULA = (
    "avg_severity ~ "
    "np.log(bike_value) + np.log(bike_cc) + C(region)"
)

FEATURE_COLS = [
    "age", "years_licensed", "bike_cc", "bike_age",
    "annual_mileage", "ncb", "region", "occupation",
]


def fit_frequency_glm(train: pd.DataFrame) -> sm.GLM:
    """Fit a Poisson GLM for claim frequency with log-exposure offset.

    Parameters
    ----------
    train:
        Training data.  Must contain all columns referenced in
        ``FREQ_FORMULA`` plus ``exposure``.

    Returns
    -------
    statsmodels fitted GLM result.
    """
    result = smf.glm(
        formula=FREQ_FORMULA,
        data=train,
        family=sm.families.Poisson(),
        offset=np.log(train["exposure"]),
    ).fit(disp=False)
    return result


def fit_severity_glm(train: pd.DataFrame) -> sm.GLM:
    """Fit a Gamma GLM for average claim severity.

    Only rows with at least one claim are used.

    Parameters
    ----------
    train:
        Training data with ``avg_severity`` and ``claim_count`` columns.

    Returns
    -------
    statsmodels fitted GLM result.
    """
    claims_only = train[train["claim_count"] > 0].copy()
    result = smf.glm(
        formula=SEV_FORMULA,
        data=claims_only,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit(disp=False)
    return result


def fit_conditional_freq_sev(
    train: pd.DataFrame,
    freq_glm: sm.GLM,
    sev_glm: sm.GLM,
) -> ConditionalFreqSev:
    """Fit the Garrido et al. conditional frequency-severity model.

    Captures frequency-severity dependence by including the expected
    claim count as a covariate in the severity model.

    Parameters
    ----------
    train:
        Full training data.
    freq_glm:
        Fitted frequency GLM (Poisson).
    sev_glm:
        Fitted base severity GLM (Gamma, claims-only).

    Returns
    -------
    Fitted :class:`~insurance_frequency_severity.ConditionalFreqSev`.
    """
    model = ConditionalFreqSev(
        freq_glm=freq_glm,
        sev_glm_base=sev_glm,
        n_as_indicator=False,
    )
    data_with_sev = train[train["claim_count"] > 0].copy()
    model.fit(
        data=data_with_sev,
        n_col="claim_count",
        s_col="avg_severity",
        sev_feature_cols=["bike_value", "bike_cc", "region"],
        freq_X=data_with_sev[FEATURE_COLS],
        exposure_col="exposure",
    )
    return model


def predict_pure_premium(
    freq_glm: sm.GLM,
    sev_glm: sm.GLM,
    data: pd.DataFrame,
) -> np.ndarray:
    """Predict pure premium = E[N|x] × E[S|x].

    Parameters
    ----------
    freq_glm:
        Fitted Poisson frequency model.
    sev_glm:
        Fitted Gamma severity model.
    data:
        Data to score.

    Returns
    -------
    np.ndarray of predicted pure premiums.
    """
    freq_pred = freq_glm.predict(data)
    sev_pred = sev_glm.predict(data)
    return freq_pred * sev_pred
