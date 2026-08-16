"""
01_frequency_severity.py
========================
Fit Poisson frequency and Gamma severity GLMs for the motorcycle portfolio,
then apply a Sarmanov copula correction for the NCD-driven negative dependence
between claim count and average severity.

Steps
-----
1. Fit a Negative-Binomial (NB2) frequency GLM on training years (2021-2022).
2. Fit a Gamma severity GLM on claim rows only.
3. Test for count-severity dependence using DependenceTest.
4. Fit the JointFreqSev Sarmanov model on top of the marginals.
5. Inspect correction factors and compare pure premiums.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from insurance_frequency_severity import DependenceTest, JointFreqSev, JointModelReport

from mc_pricing.data_generation import generate_full_portfolio

pd.set_option("display.float_format", "{:,.4f}".format)

# ---------------------------------------------------------------------------
# 1. Load data and split train / test
# ---------------------------------------------------------------------------

def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = generate_full_portfolio()

    # One-hot encode region (London as reference)
    df = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=float)

    train = df[df["calendar_year"].isin([2021, 2022])].copy()
    test  = df[df["calendar_year"] == 2023].copy()

    return train, test


# ---------------------------------------------------------------------------
# 2. Frequency GLM — Poisson with log-exposure offset
# ---------------------------------------------------------------------------

FREQ_FEATURES = [
    "rider_age", "ncd_years", "engine_cc", "vehicle_age",
    "region_Midlands", "region_North", "region_Scotland", "region_South East",
]


def fit_freq_glm(train: pd.DataFrame) -> sm.GLM:
    X = sm.add_constant(train[FREQ_FEATURES].astype(float))
    glm = sm.GLM(
        train["claim_count"],
        X,
        family=sm.families.Poisson(),
        exposure=train["exposure"].values,
    ).fit(disp=False)
    return glm


# ---------------------------------------------------------------------------
# 3. Severity GLM — Gamma, on claim rows only
# ---------------------------------------------------------------------------

def fit_sev_glm(train: pd.DataFrame) -> sm.GLM:
    claims = train[train["claim_count"] > 0].copy()
    X = sm.add_constant(claims[FREQ_FEATURES].astype(float))
    glm = sm.GLM(
        claims["avg_severity"],
        X,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit(disp=False)
    return glm


# ---------------------------------------------------------------------------
# 4. Dependence test
# ---------------------------------------------------------------------------

def run_dependence_test(train: pd.DataFrame) -> None:
    claims = train[train["claim_count"] > 0]
    test = DependenceTest()
    test.fit(
        n=claims["claim_count"].values,
        s=claims["avg_severity"].values,
    )
    print("\n=== Dependence Test (claims only) ===")
    print(test.summary())


# ---------------------------------------------------------------------------
# 5. Sarmanov joint model
# ---------------------------------------------------------------------------

def fit_joint_model(
    train: pd.DataFrame,
    freq_glm: sm.GLM,
    sev_glm: sm.GLM,
) -> JointFreqSev:
    joint = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm, copula="sarmanov")
    joint.fit(
        train,
        n_col="claim_count",
        s_col="avg_severity",
        exposure_col="exposure",
    )
    print(f"\nomega = {joint.omega_:.4f}  "
          f"95% CI = [{joint.omega_ci_[0]:.4f}, {joint.omega_ci_[1]:.4f}]")
    print(f"AIC = {joint.aic_:.2f}   BIC = {joint.bic_:.2f}")
    return joint


# ---------------------------------------------------------------------------
# 6. Compare pure premiums on test set
# ---------------------------------------------------------------------------

def compare_premiums(
    test: pd.DataFrame,
    freq_glm: sm.GLM,
    sev_glm: sm.GLM,
    joint: JointFreqSev,
) -> pd.DataFrame:
    X_test = sm.add_constant(test[FREQ_FEATURES].astype(float))
    mu_freq = freq_glm.predict(X_test) / test["exposure"].values
    mu_sev  = sev_glm.predict(X_test)

    corrections = joint.premium_correction()

    result = test[["policy_id", "exposure", "claim_count", "avg_severity"]].copy()
    result["pp_independent"] = mu_freq * mu_sev
    result["correction_factor"] = corrections["correction_factor"].values[: len(test)]
    result["pp_joint"] = result["pp_independent"] * result["correction_factor"]
    result["actual_pp"] = (result["claim_count"] * result["avg_severity"]) / result["exposure"]

    print("\n=== Pure Premium Comparison (2023 test year) ===")
    summary = result[["pp_independent", "pp_joint", "actual_pp"]].describe().T
    print(summary)

    lift = (result["pp_joint"].mean() - result["pp_independent"].mean()) / result["pp_independent"].mean()
    print(f"\nMean premium lift from copula correction: {lift:+.2%}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> tuple[sm.GLM, sm.GLM, JointFreqSev, pd.DataFrame, pd.DataFrame]:
    train, test = load_and_prepare()

    print(f"Train: {len(train):,} policies | Test: {len(test):,} policies")
    print(f"Train claim frequency: {train['claim_count'].sum()/train['exposure'].sum():.3%}")

    freq_glm = fit_freq_glm(train)
    sev_glm  = fit_sev_glm(train)

    print("\n=== Frequency GLM Summary ===")
    print(freq_glm.summary2().tables[1][["Coef.", "Std.Err.", "z", "P>|z|"]])

    print("\n=== Severity GLM Summary ===")
    print(sev_glm.summary2().tables[1][["Coef.", "Std.Err.", "z", "P>|z|"]])

    run_dependence_test(train)

    joint = fit_joint_model(train, freq_glm, sev_glm)

    result = compare_premiums(test, freq_glm, sev_glm, joint)

    return freq_glm, sev_glm, joint, train, test


if __name__ == "__main__":
    main()
