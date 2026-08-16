"""
03_ensemble_pipeline.py
=======================
Ensemble the GLM and EBM frequency models, then wrap with conformal prediction
intervals to give distribution-free coverage guarantees on pure premiums.

Pipeline
--------
1. GLM pure premium  = freq_glm × sev_glm  (from 01_frequency_severity.py)
2. EBM pure premium  = freq_ebm × sev_ebm  (from 02_gam_model.py)
3. Ensemble          = 0.5 × GLM_pp + 0.5 × EBM_pp  (simple average blend)
4. Conformal wrapper = InsuranceConformalPredictor calibrated on 2022
                       gives 90% prediction intervals for 2023 test year

The conformal step is model-agnostic: we pass a thin sklearn-compatible wrapper
around the ensemble so InsuranceConformalPredictor can call .predict().
"""

import sys
sys.path.insert(0, "/home/runner/work/explore_burning_cost_mc/explore_burning_cost_mc")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin

from insurance_conformal import InsuranceConformalPredictor
from insurance_gam.ebm import InsuranceEBM

from mc_pricing.data_generation import generate_full_portfolio


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

GLM_FEATURES = [
    "rider_age", "ncd_years", "engine_cc", "vehicle_age",
    "region_Midlands", "region_North", "region_Scotland", "region_South East",
]
EBM_FEATURES = ["rider_age", "ncd_years", "engine_cc", "vehicle_age", "region"]


# ---------------------------------------------------------------------------
# Sklearn-compatible ensemble wrapper
# ---------------------------------------------------------------------------

class EnsemblePurePremium(BaseEstimator, RegressorMixin):
    """
    Weighted ensemble of GLM-based and EBM-based pure premium predictions.

    Accepts a DataFrame with all features; routes each sub-model to its
    required feature set internally.
    """

    def __init__(
        self,
        freq_glm: sm.GLM,
        sev_glm: sm.GLM,
        freq_ebm: InsuranceEBM,
        sev_ebm: InsuranceEBM,
        glm_weight: float = 0.50,
    ) -> None:
        self.freq_glm   = freq_glm
        self.sev_glm    = sev_glm
        self.freq_ebm   = freq_ebm
        self.sev_ebm    = sev_ebm
        self.glm_weight = glm_weight

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        exposure = X["exposure"].values if "exposure" in X.columns else np.ones(len(X))

        # GLM pure premium (annualised)
        X_glm = sm.add_constant(X[GLM_FEATURES].astype(float), has_constant="add")
        pp_glm = (self.freq_glm.predict(X_glm) / exposure) * self.sev_glm.predict(X_glm)

        # EBM pure premium
        pp_ebm = (
            self.freq_ebm.predict(X[EBM_FEATURES]) / exposure
            * self.sev_ebm.predict(X[EBM_FEATURES])
        )

        return self.glm_weight * pp_glm + (1 - self.glm_weight) * pp_ebm


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = generate_full_portfolio()
    df = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=float)
    # Restore region string column for EBM
    df_raw = generate_full_portfolio()
    df["region"] = df_raw["region"].values

    train = df[df["calendar_year"] == 2021].copy()
    cal   = df[df["calendar_year"] == 2022].copy()
    test  = df[df["calendar_year"] == 2023].copy()
    return train, cal, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_ensemble_with_conformal(
    freq_glm: sm.GLM,
    sev_glm: sm.GLM,
    freq_ebm: InsuranceEBM,
    sev_ebm: InsuranceEBM,
) -> tuple[EnsemblePurePremium, InsuranceConformalPredictor, pd.DataFrame]:

    train, cal, test = prepare_data()

    ensemble = EnsemblePurePremium(freq_glm, sev_glm, freq_ebm, sev_ebm)

    # Calibrate conformal predictor on 2022
    conformal = InsuranceConformalPredictor(
        model=ensemble,
        nonconformity="pearson_weighted",
        distribution="tweedie",
        tweedie_power=1.5,
    )
    actual_pp_cal = (cal["claim_count"] * cal["avg_severity"]) / cal["exposure"]
    conformal.calibrate(cal, y_cal=actual_pp_cal.values, exposure=cal["exposure"].values)

    # Predict intervals on 2023
    intervals = conformal.predict_interval(test, alpha=0.10).to_pandas()
    actual_pp_test = (test["claim_count"] * test["avg_severity"]) / test["exposure"]

    result = pd.concat([
        test[["policy_id", "exposure", "claim_count"]].reset_index(drop=True),
        intervals.reset_index(drop=True),
    ], axis=1)
    result["actual_pp"] = actual_pp_test.values

    # Empirical coverage
    covered = (
        (result["actual_pp"] >= result["lower"])
        & (result["actual_pp"] <= result["upper"])
    )
    coverage = covered.mean()
    avg_width = (result["upper"] - result["lower"]).mean()

    print("\n=== Conformal Ensemble (2023 test year) ===")
    print(conformal.summary())
    print(f"\nEmpirical coverage @ 90%: {coverage:.1%}")
    print(f"Average interval width:   £{avg_width:,.0f}")
    print(f"\nInterval sample (first 5 rows):")
    print(result[["policy_id", "lower", "point", "upper", "actual_pp"]].head().to_string(index=False))

    return ensemble, conformal, result


if __name__ == "__main__":
    # Stand-alone: refit sub-models from scratch (slow but self-contained)
    import statsmodels.api as sm
    from mc_pricing.data_generation import generate_full_portfolio

    df_raw = generate_full_portfolio()
    df = pd.get_dummies(df_raw.copy(), columns=["region"], drop_first=True, dtype=float)
    df["region"] = df_raw["region"].values

    train = df[df["calendar_year"] == 2021].copy()

    X_train_glm = sm.add_constant(train[GLM_FEATURES].astype(float))
    freq_glm = sm.GLM(train["claim_count"], X_train_glm,
                      family=sm.families.Poisson(),
                      exposure=train["exposure"].values).fit(disp=False)

    claims_train = train[train["claim_count"] > 0]
    X_cls_glm = sm.add_constant(claims_train[GLM_FEATURES].astype(float))
    sev_glm = sm.GLM(claims_train["avg_severity"], X_cls_glm,
                     family=sm.families.Gamma(link=sm.families.links.Log())).fit(disp=False)

    freq_ebm = InsuranceEBM(loss="poisson", interactions="3x")
    freq_ebm.fit(train[EBM_FEATURES], train["claim_count"].values,
                 exposure=train["exposure"].values)

    sev_ebm = InsuranceEBM(loss="gamma")
    sev_ebm.fit(claims_train[EBM_FEATURES], claims_train["avg_severity"].values)

    build_ensemble_with_conformal(freq_glm, sev_glm, freq_ebm, sev_ebm)
