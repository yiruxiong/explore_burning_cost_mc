"""
02_gam_model.py
===============
Fit Explainable Boosting Machine (EBM) frequency and severity models using
insurance-gam. EBMs provide non-linear shape functions per rating factor that
a pricing actuary can inspect and challenge — no post-hoc SHAP required.

Steps
-----
1. Fit an EBM frequency model (Poisson) on 2021-2022 training data.
2. Fit an EBM severity model (Gamma) on claim rows only.
3. Print RelativitiesTable for each key rating factor.
4. Compare Gini coefficients between GLM and EBM predictions on test year.
"""

import sys
sys.path.insert(0, "/home/runner/work/explore_burning_cost_mc/explore_burning_cost_mc")

import numpy as np
import pandas as pd
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

from mc_pricing.data_generation import generate_full_portfolio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURES = [
    "rider_age", "ncd_years", "engine_cc", "vehicle_age",
]

# We avoid one-hot encoding for EBMs — they handle categoricals natively,
# so we pass region as a string column.
FEATURES_WITH_REGION = FEATURES + ["region"]


def _gini(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    """Exposure-weighted Gini coefficient (double-lift style)."""
    order = np.argsort(y_pred)
    y_sorted = y_true[order]
    w_sorted = weight[order]
    cum_loss = np.cumsum(y_sorted * w_sorted) / (y_sorted * w_sorted).sum()
    cum_exp  = np.cumsum(w_sorted) / w_sorted.sum()
    gini = 2 * np.trapezoid(cum_loss, cum_exp) - 1
    return gini


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fit_ebm_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[InsuranceEBM, InsuranceEBM]:

    # --- Frequency EBM ---
    freq_ebm = InsuranceEBM(loss="poisson", interactions="3x")
    freq_ebm.fit(
        train[FEATURES_WITH_REGION],
        train["claim_count"].values,
        exposure=train["exposure"].values,
    )

    # --- Severity EBM (on claim rows only) ---
    claims_train = train[train["claim_count"] > 0].copy()
    sev_ebm = InsuranceEBM(loss="gamma")
    sev_ebm.fit(
        claims_train[FEATURES_WITH_REGION],
        claims_train["avg_severity"].values,
    )

    return freq_ebm, sev_ebm


def print_relativities(freq_ebm: InsuranceEBM, sev_ebm: InsuranceEBM) -> None:
    freq_rt = RelativitiesTable(freq_ebm)
    sev_rt  = RelativitiesTable(sev_ebm)

    print("\n=== Frequency EBM — Rider Age relativities ===")
    print(freq_rt.table("rider_age").head(20))

    print("\n=== Frequency EBM — NCD Years relativities ===")
    print(freq_rt.table("ncd_years"))

    print("\n=== Frequency EBM — Engine CC relativities ===")
    print(freq_rt.table("engine_cc").head(10))

    print("\n=== Frequency EBM — Region relativities ===")
    print(freq_rt.table("region"))

    print("\n=== Severity EBM — Engine CC relativities ===")
    print(sev_rt.table("engine_cc").head(10))

    print("\n=== Frequency EBM Summary ===")
    print(freq_rt.summary())


def evaluate(
    test: pd.DataFrame,
    freq_ebm: InsuranceEBM,
    sev_ebm: InsuranceEBM,
) -> pd.DataFrame:

    mu_freq = freq_ebm.predict(test[FEATURES_WITH_REGION]) / test["exposure"].values
    mu_sev  = sev_ebm.predict(test[FEATURES_WITH_REGION])

    result = test[["policy_id", "exposure", "claim_count", "avg_severity"]].copy()
    result["pred_freq_ebm"] = mu_freq
    result["pred_sev_ebm"]  = mu_sev
    result["pred_pp_ebm"]   = mu_freq * mu_sev

    actual_pp = (test["claim_count"] * test["avg_severity"]) / test["exposure"]
    g = _gini(actual_pp.values, result["pred_pp_ebm"].values, test["exposure"].values)
    print(f"\nEBM pure premium Gini (2023 test): {g:.4f}")

    return result


def main() -> tuple[InsuranceEBM, InsuranceEBM, pd.DataFrame]:
    df = generate_full_portfolio()
    # EBMs handle region as a string — no encoding needed
    train = df[df["calendar_year"].isin([2021, 2022])].copy()
    test  = df[df["calendar_year"] == 2023].copy()

    print(f"Fitting EBMs on {len(train):,} training policies …")
    freq_ebm, sev_ebm = fit_ebm_models(train, test)

    print_relativities(freq_ebm, sev_ebm)

    result = evaluate(test, freq_ebm, sev_ebm)
    return freq_ebm, sev_ebm, result


if __name__ == "__main__":
    main()
