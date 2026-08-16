"""
08_fairness_governance.py
=========================
Fairness audit and model risk management documentation for the motorcycle
pricing model.

Part A — Fairness Audit (insurance-fairness)
---------------------------------------------
Checks for demographic parity violations and proxy discrimination across
two protected characteristics that are relevant under UK Equality Act 2010:
  - rider_age_band (age as a protected characteristic)
  - region (as a potential proxy for ethnicity / socio-economic status)

Part B — Model Risk Card (insurance-governance)
------------------------------------------------
Produces a structured MRMModelCard for the ensemble pricing model, recording
materiality tier, training data period, monitoring frequency and regulatory use.
"""

import sys
sys.path.insert(0, "/home/runner/work/explore_burning_cost_mc/explore_burning_cost_mc")

import numpy as np
import pandas as pd
import statsmodels.api as sm

from insurance_fairness import FairnessAudit
from insurance_governance import MRMModelCard

from mc_pricing.data_generation import generate_full_portfolio


GLM_FEATURES = [
    "rider_age", "ncd_years", "engine_cc", "vehicle_age",
    "region_Midlands", "region_North", "region_Scotland", "region_South East",
]


# ---------------------------------------------------------------------------
# Prepare data
# ---------------------------------------------------------------------------

def prepare_audit_data() -> tuple[pd.DataFrame, sm.GLM]:
    df_raw = generate_full_portfolio()
    df = pd.get_dummies(df_raw.copy(), columns=["region"], drop_first=True, dtype=float)
    df["region"] = df_raw["region"].values

    # Age band (grouping for fairness audit — not used as a rating factor)
    df["age_band"] = pd.cut(
        df["rider_age"],
        bins=[16, 25, 35, 50, 65, 80],
        labels=["17-25", "26-35", "36-50", "51-65", "66+"],
    ).astype(str)

    train = df[df["calendar_year"].isin([2021, 2022])].copy()
    test  = df[df["calendar_year"] == 2023].copy()

    X_tr = sm.add_constant(train[GLM_FEATURES].astype(float))
    glm = sm.GLM(
        train["claim_count"], X_tr,
        family=sm.families.Poisson(),
        exposure=train["exposure"].values,
    ).fit(disp=False)

    X_te = sm.add_constant(test[GLM_FEATURES].astype(float), has_constant="add")
    test["prediction"] = glm.predict(X_te)
    test["actual_pp"] = (test["claim_count"] * test["avg_severity"]) / test["exposure"]

    return test, glm


# ---------------------------------------------------------------------------
# Part A: Fairness Audit
# ---------------------------------------------------------------------------

def run_fairness_audit(test: pd.DataFrame, glm) -> None:
    audit = FairnessAudit(
        model=glm,
        data=test,
        protected_cols=["age_band", "region"],
        prediction_col="prediction",
        outcome_col="claim_count",
        exposure_col="exposure",
        factor_cols=["rider_age", "ncd_years", "engine_cc", "vehicle_age"],
        model_name="UK Motorcycle Ensemble Pricing Model",
        run_proxy_detection=True,
        run_counterfactual=False,
        n_bootstrap=0,        # set >0 for bootstrap CIs (slow)
    )

    report = audit.run()

    print("\n=== Fairness Audit ===")
    print(f"Overall RAG status: {report.overall_rag}")

    for pc, pc_report in report.results.items():
        print(f"\n--- Protected characteristic: {pc} ---")
        dp = pc_report.demographic_parity
        print(f"  Demographic parity RAG:  {dp.rag}")
        print(f"  Max log-premium ratio:   {dp.log_ratio:.4f}")
        cal = pc_report.calibration
        if cal is not None:
            print(f"  Calibration RAG:         {cal.rag}")

    return report


# ---------------------------------------------------------------------------
# Part B: MRM Model Card
# ---------------------------------------------------------------------------

def create_model_card() -> MRMModelCard:
    from insurance_governance import Assumption

    assumptions = [
        Assumption(
            description=(
                "Sarmanov omega estimated on 2021-2022 data — may shift if NCD "
                "accumulation behaviour changes post-COVID."
            ),
            risk="HIGH",
            mitigation="Annual re-estimation of omega; monitored via DependenceTest.",
            rationale="NCD claim suppression is behavioural and can change with market conditions.",
        ),
        Assumption(
            description=(
                "EBM region encoding treats region as nominal; postcode-level "
                "granularity not captured."
            ),
            risk="MEDIUM",
            mitigation="Plan to introduce postcode sector bands in next model iteration.",
            rationale="Regional aggregation reduces overfitting but misses local price signals.",
        ),
        Assumption(
            description="Frequency model does not account for inflationary trend in repair costs.",
            risk="MEDIUM",
            mitigation="Annual severity trend analysis; manual CPI uplift applied at rating.",
            rationale="Severity GLM fitted on nominal costs; UK vehicle repair inflation ~8% pa.",
        ),
    ]

    card = MRMModelCard(
        model_id="MC-PRICING-001",
        model_name="UK Motorcycle Ensemble Pricing Model",
        version="1.0.0",
        model_class="pricing",
        model_type="Ensemble (Poisson-GLM + EBM) with Sarmanov copula",
        intended_use=(
            "Technical pricing of UK personal lines motorcycle insurance. "
            "Produces risk-adjusted pure premium for quote generation."
        ),
        not_intended_for=["Commercial fleet", "Track-day / competition use"],
        target_variable="claim_frequency × average_severity (pure premium £)",
        distribution_family="Poisson × Gamma with Sarmanov copula correction",
        rating_factors=["rider_age", "ncd_years", "engine_cc", "vehicle_age", "region"],
        training_data_period=("2021-01-01", "2022-12-31"),
        development_date="2024-01-01",
        developer="Pricing Team",
        champion_challenger_status="champion",
        assumptions=assumptions,
        limitations=[],
        outstanding_issues=[],
        portfolio_scope="Motorcycle — private use",
        geographic_scope="United Kingdom",
        customer_facing=False,
        regulatory_use=True,
        gwp_impacted=8_000_000.0,
        materiality_tier=2,
        tier_rationale=(
            "Directly impacts policyholder pricing. Model output used for "
            "rated quotes. Requires annual independent validation under PRA SS3/18."
        ),
        approved_by=[],
        approval_date="",
        approval_conditions="",
        next_review_date="2025-01-01",
        monitoring_owner="Actuarial Pricing",
        monitoring_frequency="Quarterly",
        monitoring_triggers={"gini_drift_p": 0.05, "ae_ratio_lower": 0.85, "psi": 0.25},
        trigger_actions={
            "gini_drift_p": "Initiate model refit review",
            "ae_ratio_lower": "Apply balance recalibration",
            "psi": "Investigate population shift",
        },
        last_monitoring_run="",
        last_validation_run="",
        last_validation_run_id="",
        overall_rag="amber",
        created_at="2024-01-01",
        updated_at="2024-01-01",
    )

    print("\n=== MRM Model Card ===")
    print(f"Model:             {card.model_name}  v{card.version}")
    print(f"Type:              {card.model_type}")
    print(f"Materiality tier:  {card.materiality_tier}")
    print(f"Training period:   {card.training_data_period[0]} – {card.training_data_period[1]}")
    print(f"Monitoring:        {card.monitoring_frequency} — owner: {card.monitoring_owner}")
    print(f"Next review:       {card.next_review_date}")
    print(f"GWP impacted:      £{card.gwp_impacted:,.0f}")
    print(f"Overall RAG:       {card.overall_rag}")
    print(f"Assumptions ({len(card.assumptions)}):")
    for a in card.assumptions:
        print(f"  [{a.risk.upper()}] {a.description}")

    return card


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    test, glm = prepare_audit_data()
    run_fairness_audit(test, glm)
    create_model_card()


if __name__ == "__main__":
    main()
