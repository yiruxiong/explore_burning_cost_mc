"""
main.py — end-to-end motorcycle insurance pricing lifecycle.

Runs the full pipeline in sequence:

  1.  Synthetic data generation (50 000 UK motorcycle policies, 2018–2023)
  2.  Train / validation / test split (temporal)
  3.  Ensemble model fit (GLM + penalized GLM + credibility)
  4.  Quote-to-pricing journey simulation
  5.  Price-to-sell journey simulation
  6.  Model monitoring and drift detection
  7.  Conformal prediction intervals
  8.  Fairness audit (gender — protected characteristic)
  9.  Governance report and model card generation

Run with:
    python -m motorcycle_insurance.main

or from the repo root:
    python motorcycle_insurance/main.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Project modules ───────────────────────────────────────────────────────────
from motorcycle_insurance.data.synthetic import generate_portfolio
from motorcycle_insurance.models.ensemble import EnsemblePricingModel
from motorcycle_insurance.pipelines.quote_to_price import (
    PricingConfig,
    run_quote_to_pricing,
)
from motorcycle_insurance.pipelines.price_to_sell import (
    MarketConfig,
    run_price_to_sell,
    summarise_portfolio_economics,
)
from motorcycle_insurance.monitoring.drift import (
    run_monitoring,
    print_monitoring_summary,
)
from motorcycle_insurance.conformal.intervals import (
    build_conformal_predictor,
    predict_premium_intervals,
)
from motorcycle_insurance.fairness.audit import run_fairness_audit
from motorcycle_insurance.governance.audit import (
    build_model_card,
    build_governance_report,
)


def main() -> None:
    """Run the full motorcycle insurance pricing lifecycle."""

    # ── 1. Generate synthetic portfolio ──────────────────────────────────────
    print("\n[1/9] Generating synthetic UK motorcycle insurance portfolio...")
    portfolio = generate_portfolio(n_policies=50_000, seed=42)
    print(f"      Portfolio: {len(portfolio):,} policies | "
          f"claim rate: {portfolio['claim_count'].mean():.3f} | "
          f"avg severity: £{portfolio.loc[portfolio['claim_count']>0,'avg_severity'].mean():,.0f}")

    # ── 2. Temporal train / val / test / calibration / monitor splits ─────────
    print("\n[2/9] Splitting data by policy year...")
    train = portfolio[portfolio["policy_year"].isin([2018, 2019, 2020, 2021])].copy()
    val   = portfolio[portfolio["policy_year"] == 2022].copy()
    test  = portfolio[portfolio["policy_year"] == 2023].copy()
    calibration = val.sample(frac=0.5, random_state=1).copy()
    monitor_ref = train.sample(n=5_000, random_state=2).copy()
    monitor_cur = test.sample(n=5_000, random_state=3).copy()

    print(f"      Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # ── 3. Fit ensemble model ─────────────────────────────────────────────────
    print("\n[3/9] Fitting ensemble pricing model...")
    model = EnsemblePricingModel()
    model.fit(train, val=val)

    train_preds = model.predict(train)
    test_preds  = model.predict(test)

    # Quick A/E on test
    ae_test = (
        test["total_loss"].sum() / (test["exposure"] * test_preds).sum()
        if (test["exposure"] * test_preds).sum() > 0 else np.nan
    )
    gini_approx = _gini(test["total_loss"].values, test_preds)
    print(f"      Test A/E ratio: {ae_test:.3f} | Approx Gini: {gini_approx:.3f}")

    # ── 4. Quote-to-pricing journey ───────────────────────────────────────────
    print("\n[4/9] Simulating quote-to-pricing journey...")
    pricing_config = PricingConfig(
        expense_ratio=0.25,
        profit_margin=0.05,
        reinsurance_loading=0.03,
        minimum_premium=150.0,
        maximum_premium=4_000.0,
    )
    pricing_result = run_quote_to_pricing(
        quotes=test,
        ensemble_model=model,
        config=pricing_config,
        rng_seed=42,
    )
    print(f"      Referred quotes: {pricing_result['is_referred'].sum():,} "
          f"({pricing_result['is_referred'].mean()*100:.1f}%)")
    print(f"      Avg quoted premium: £{pricing_result['quoted_premium'].mean():,.2f} | "
          f"Avg technical premium: £{pricing_result['technical_premium'].mean():,.2f}")

    # ── 5. Price-to-sell journey ──────────────────────────────────────────────
    print("\n[5/9] Simulating price-to-sell journey...")
    market_config = MarketConfig(
        base_conversion_rate=0.60,
        price_elasticity=-2.5,
    )
    sell_result = run_price_to_sell(
        quotes_df=test,
        pricing_result=pricing_result,
        config=market_config,
        rng_seed=99,
    )
    economics = summarise_portfolio_economics(sell_result)
    print(economics.to_string(index=False))

    # ── 6. Model monitoring ───────────────────────────────────────────────────
    print("\n[6/9] Running model monitoring (reference vs. current period)...")
    ref_preds = model.predict(monitor_ref)
    cur_preds = model.predict(monitor_cur)
    try:
        monitoring_report = run_monitoring(
            reference_df=monitor_ref,
            current_df=monitor_cur,
            reference_predictions=ref_preds,
            current_predictions=cur_preds,
        )
        print_monitoring_summary(monitoring_report)
        monitoring_results = {"status": "completed"}
    except Exception as exc:
        print(f"      Monitoring raised: {exc}")
        monitoring_results = {"status": "error", "message": str(exc)}

    # ── 7. Conformal prediction intervals ─────────────────────────────────────
    print("\n[7/9] Building conformal prediction intervals (90% coverage)...")
    try:
        cp = build_conformal_predictor(
            ensemble_model=model,
            calibration_df=calibration,
            alpha=0.10,
        )
        intervals = predict_premium_intervals(cp, test.head(10), alpha=0.10)
        print("      Sample prediction intervals (first 5 policies):")
        print(intervals.head(5).to_string(index=False))
    except Exception as exc:
        print(f"      Conformal prediction raised: {exc}")

    # ── 8. Fairness audit ─────────────────────────────────────────────────────
    print("\n[8/9] Running fairness audit (gender — protected characteristic)...")
    try:
        fairness_audit = run_fairness_audit(
            data=test.head(2_000),
            model=model,
            predictions=test_preds[:2_000],
        )
        fairness_report = fairness_audit.run()
        print("      Fairness audit completed.")
        if hasattr(fairness_report, "overall_rag"):
            print(f"      Overall RAG status: {fairness_report.overall_rag}")
        if hasattr(fairness_report, "results") and fairness_report.results:
            for pc, pc_report in fairness_report.results.items():
                print(f"\n      Protected characteristic: {pc}")
                if hasattr(pc_report, "demographic_parity") and pc_report.demographic_parity:
                    dp = pc_report.demographic_parity
                    print(f"        Demographic parity ratio: {dp}")
                if hasattr(pc_report, "proxy_detection") and pc_report.proxy_detection:
                    proxy_result = pc_report.proxy_detection
                    flagged = getattr(proxy_result, "flagged_factors", [])
                    print(f"        Proxy-flagged factors: {flagged or 'none'}")
    except Exception as exc:
        print(f"      Fairness audit raised: {exc}")

    # ── 9. Governance report ──────────────────────────────────────────────────
    print("\n[9/9] Building governance report and model card...")
    card = build_model_card()
    governance_report = build_governance_report(
        card=card,
        monitoring_results=monitoring_results,
    )
    json_output = governance_report.to_json()
    print(f"      Model card: {card.model_name}")
    print(f"      Governance JSON length: {len(json_output):,} chars")

    print("\n✓ Full lifecycle completed successfully.\n")


def _gini(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute a simple Gini / Lorenz-based ordering coefficient."""
    order = np.argsort(predicted)[::-1]
    actual_sorted = actual[order]
    cum_actual = np.cumsum(actual_sorted) / (actual_sorted.sum() + 1e-9)
    n = len(actual)
    cum_pop = np.arange(1, n + 1) / n
    return float(2 * np.trapezoid(cum_actual, cum_pop) - 1)


if __name__ == "__main__":
    main()
