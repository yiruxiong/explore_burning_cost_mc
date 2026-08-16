"""
07_monitoring.py
================
Production monitoring for the motorcycle insurance pricing models.

Monitors whether the ensemble model predictions remain valid as new calendar
years are observed. Uses insurance-monitoring's ModelMonitor to track:

- Gini coefficient stability (discrimination drift)
- PSI / CSI population shift
- A/E (Actual vs Expected) calibration ratio with confidence intervals

Reference year: 2021 (training baseline).
Monitoring years: 2022, 2023.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from insurance_monitoring import ModelMonitor, MonitoringReport, ae_ratio, ae_ratio_ci, psi

from mc_pricing.data_generation import generate_full_portfolio


GLM_FEATURES = [
    "rider_age", "ncd_years", "engine_cc", "vehicle_age",
    "region_Midlands", "region_North", "region_Scotland", "region_South East",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=float)
    for col in GLM_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
    return out


def fit_glm(train: pd.DataFrame) -> sm.GLM:
    X = sm.add_constant(train[GLM_FEATURES].astype(float))
    return sm.GLM(
        train["claim_count"], X,
        family=sm.families.Poisson(),
        exposure=train["exposure"].values,
    ).fit(disp=False)


def get_predictions(df: pd.DataFrame, glm: sm.GLM) -> np.ndarray:
    X = sm.add_constant(df[GLM_FEATURES].astype(float), has_constant="add")
    return glm.predict(X)


# ---------------------------------------------------------------------------
# Gini monitoring via ModelMonitor
# ---------------------------------------------------------------------------

def run_model_monitor(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    glm: sm.GLM,
) -> None:
    ref_preds  = get_predictions(ref_df, glm)
    curr_preds = get_predictions(curr_df, glm)

    # Actual claim count is our observed outcome; predictions are expected count
    monitor = ModelMonitor(distribution="poisson", n_bootstrap=200, random_state=42)
    monitor.fit(
        y_ref=ref_df["claim_count"].values,
        y_hat_ref=ref_preds,
        exposure_ref=ref_df["exposure"].values,
    )

    result = monitor.test(
        y_new=curr_df["claim_count"].values,
        y_hat_new=curr_preds,
        exposure_new=curr_df["exposure"].values,
    )

    print(f"\n=== Model Monitor: {ref_df['calendar_year'].iloc[0]} → "
          f"{curr_df['calendar_year'].iloc[0]} ===")
    print(result.summary())


# ---------------------------------------------------------------------------
# A/E calibration monitoring
# ---------------------------------------------------------------------------

def run_ae_monitoring(df: pd.DataFrame, glm: sm.GLM, year: int) -> None:
    pred = get_predictions(df, glm)
    actual   = df["claim_count"].values
    expected = pred  # model predicts expected claim count (with exposure offset)

    ae = ae_ratio(actual, expected)
    ci = ae_ratio_ci(actual, expected, alpha=0.05)
    lo, hi = ci["lower"], ci["upper"]

    print(f"\n=== A/E Calibration ({year}) ===")
    print(f"A/E ratio: {ae:.4f}  95% CI: [{lo:.4f}, {hi:.4f}]")
    if lo <= 1.0 <= hi:
        print("✓ Model is well-calibrated (1.0 within CI).")
    elif ae > 1.0:
        print("⚠ Model under-predicts — consider upward recalibration.")
    else:
        print("⚠ Model over-predicts — consider downward recalibration.")


# ---------------------------------------------------------------------------
# PSI feature drift
# ---------------------------------------------------------------------------

def run_psi(ref_df: pd.DataFrame, curr_df: pd.DataFrame, feature: str, year: int) -> None:
    psi_val = psi(ref_df[feature].values, curr_df[feature].values, n_bins=10)
    flag = "🟢 stable" if psi_val < 0.10 else ("🟡 slight shift" if psi_val < 0.25 else "🔴 major shift")
    print(f"  PSI({feature}, {year}): {psi_val:.4f}  {flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = generate_full_portfolio()
    df_enc = encode(df.copy())
    df_enc["region"]        = df["region"].values
    df_enc["calendar_year"] = df["calendar_year"].values

    ref  = df_enc[df_enc["calendar_year"] == 2021].copy()
    yr22 = df_enc[df_enc["calendar_year"] == 2022].copy()
    yr23 = df_enc[df_enc["calendar_year"] == 2023].copy()

    glm = fit_glm(ref)

    # --- Gini monitoring ---
    run_model_monitor(ref, yr22, glm)
    run_model_monitor(ref, yr23, glm)

    # --- A/E monitoring ---
    run_ae_monitoring(yr22, glm, 2022)
    run_ae_monitoring(yr23, glm, 2023)

    # --- PSI feature drift ---
    print("\n=== PSI Feature Drift (vs 2021 reference) ===")
    for yr_df, yr in [(yr22, 2022), (yr23, 2023)]:
        for feat in ["rider_age", "engine_cc", "ncd_years"]:
            run_psi(ref, yr_df, feat, yr)


if __name__ == "__main__":
    main()
