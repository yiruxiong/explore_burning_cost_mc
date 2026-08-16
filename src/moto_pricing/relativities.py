"""
Actuarial-grade rating relativities from the CatBoost frequency model.

Wraps ``burning_cost.shap_relativities.SHAPRelativities`` — this is the
package's headline technique: turn a black-box GBM's SHAP values into the
same (feature, level, multiplicative relativity) table a GLM's exp(beta)
coefficients would give you, so it can go straight into a rating engine and
be reviewed by a pricing committee the same way a GLM factor table would be.
"""

from __future__ import annotations

import pandas as pd
from burning_cost.shap_relativities import SHAPRelativities

from moto_pricing.features import FREQUENCY_FEATURES, CATEGORICAL_FEATURES

# One sensible, lowest-risk base level per categorical rating factor.
BASE_LEVELS: dict[str, str] = {
    "bike_type": "Tourer",
    "security": "alarm_and_tracker",
    "overnight_parking": "garage",
    "area": "A",
    "policy_type": "Comp",
}


def extract_frequency_relativities(gbm_frequency_model, X: pd.DataFrame, exposure: pd.Series) -> pd.DataFrame:
    """
    SHAP-based rating relativities for the reference CatBoost frequency model.

    Parameters
    ----------
    gbm_frequency_model
        A fitted CatBoost model trained with a Poisson objective and a
        ``log(exposure)`` baseline — e.g. ``GBMPricingModel.frequency_model``.
    X : pd.DataFrame
        Feature matrix the model was trained on (``FREQUENCY_FEATURES`` columns).
    exposure : pd.Series
        Earned exposure, used as observation weights.

    Returns
    -------
    pd.DataFrame
        Columns: feature, level, relativity, lower_ci, upper_ci, mean_shap,
        shap_std, n_obs, exposure_weight. One row per (feature, level).
    """
    cat_features = [f for f in FREQUENCY_FEATURES if f in CATEGORICAL_FEATURES]
    sr = SHAPRelativities(
        gbm_frequency_model, X[FREQUENCY_FEATURES], exposure=exposure,
        categorical_features=cat_features,
    )
    sr.fit()
    checks = sr.validate()
    if not checks["reconstruction"].passed:
        raise RuntimeError(f"SHAP reconstruction check failed: {checks['reconstruction'].message}")

    base_levels = {f: v for f, v in BASE_LEVELS.items() if f in cat_features}
    return sr.extract_relativities(normalise_to="base_level", base_levels=base_levels)
