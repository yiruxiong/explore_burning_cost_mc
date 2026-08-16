"""
06_price_to_sell.py
===================
Simulate the price-to-sell (pricing → conversion) journey.

In UK personal lines, a technical price is not the price offered — the insurer
must decide whether to apply a discount or load based on competitive position.
This script simulates:

1. Generate a synthetic conversion dataset where a random A/B discount test
   was run: group A (control) received the model price; group B (treated)
   received a 10% discount. Conversion to policy is a logistic function of
   the affordability gap (market price − offered price).

2. Use CausalPricingModel (Double ML) to estimate the causal effect of the
   discount on conversion rate.

3. Compute uplift: expected additional GWP vs. cost of discount, to decide
   whether the 10% discount pays for itself through increased volume.

This mirrors a real UK pricing team's A/B testing and causal inference workflow.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import BinaryTreatment

RNG = np.random.default_rng(77)


# ---------------------------------------------------------------------------
# Simulate A/B test conversion data
# ---------------------------------------------------------------------------

def simulate_conversion_data(n: int = 8_000) -> pd.DataFrame:
    """
    Generate synthetic quote conversion data from a 10% discount A/B test.

    Treatment = 1 if the quote received a 10% discount, else 0.
    Outcome   = 1 if the policyholder converted (bought the policy).

    True causal effect: discount increases conversion probability by ~8 pp
    in the high-price segment and ~3 pp in the low-price segment.
    """
    # Confounders (observable risk characteristics)
    rider_age   = RNG.integers(17, 76, n)
    engine_cc   = RNG.choice([125, 250, 400, 600, 800, 1000, 1200], n)
    ncd_years   = RNG.choice([0, 1, 2, 3, 4], n)
    region_idx  = RNG.choice([0, 1, 2, 3, 4], n, p=[0.30, 0.25, 0.20, 0.15, 0.10])

    # Base conversion probability (affected by price sensitivity, age, etc.)
    base_conv = 0.35 + 0.004 * (ncd_years) - 0.001 * np.abs(rider_age - 35) + 0.02 * (engine_cc < 400)

    # Treatment: random assignment (50/50 discount vs. control)
    treatment = RNG.binomial(1, 0.50, n)

    # True causal effect of discount: larger for price-sensitive segments
    price_sensitivity = 0.08 + 0.04 * (engine_cc < 400).astype(float) - 0.02 * ncd_years
    true_effect = price_sensitivity * treatment

    # Conversion outcome
    conv_prob = np.clip(base_conv + true_effect + RNG.normal(0, 0.05, n), 0.01, 0.99)
    converted = RNG.binomial(1, conv_prob, n)

    # Annualised pure premium (proxy for price)
    base_pp = 280 + 1.2 * engine_cc / 10 + 40 * np.maximum(0, 3 - ncd_years)
    pure_premium = base_pp * (1 + 0.08 * region_idx / 4)

    return pd.DataFrame({
        "rider_age":     rider_age,
        "engine_cc":     engine_cc,
        "ncd_years":     ncd_years,
        "region_idx":    region_idx,
        "pure_premium":  pure_premium,
        "discount_flag": treatment,
        "converted":     converted,
        "true_effect":   price_sensitivity,   # kept for validation only
    })


# ---------------------------------------------------------------------------
# Fit Causal Pricing Model (Double ML)
# ---------------------------------------------------------------------------

def fit_causal_model(df: pd.DataFrame) -> CausalPricingModel:
    confounders = ["rider_age", "engine_cc", "ncd_years", "region_idx", "pure_premium"]

    model = CausalPricingModel(
        outcome="converted",
        outcome_type="binary",
        treatment=BinaryTreatment(column="discount_flag"),
        confounders=confounders,
    )
    model.fit(df)
    return model


# ---------------------------------------------------------------------------
# Uplift analysis
# ---------------------------------------------------------------------------

def uplift_analysis(df: pd.DataFrame, model: CausalPricingModel) -> None:
    ate_result = model.average_treatment_effect()
    ate    = ate_result.estimate
    ci_lo  = ate_result.ci_lower
    ci_hi  = ate_result.ci_upper

    print("\n=== Causal Effect of 10% Discount on Conversion Rate ===")
    print(f"ATE (average treatment effect): {ate:+.4f}")
    print(f"95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"p-value: {ate_result.p_value:.4f}")

    # Commercial decision: does the discount pay for itself?
    avg_premium = df["pure_premium"].mean() * 1.20
    discount_cost_per_policy   = avg_premium * 0.10
    additional_revenue_per_quote = ate * avg_premium

    roi = additional_revenue_per_quote / discount_cost_per_policy - 1

    print(f"\nAverage loaded premium:          £{avg_premium:,.0f}")
    print(f"Discount cost per policy:         £{discount_cost_per_policy:,.0f}")
    print(f"Additional revenue per quote:     £{additional_revenue_per_quote:,.2f}")
    print(f"Discount ROI:                     {roi:+.1%}")

    if roi > 0:
        print("\n→ Discount is commercially justified on average.")
    else:
        print("\n→ Discount destroys value on average; consider segment targeting.")

    # Segment-level CATE by NCD band
    print("\n=== Segment CATE by NCD years ===")
    try:
        seg = model.cate_by_segment(df, segment_col="ncd_years", min_segment_size=500)
        print(seg[["segment", "cate_estimate", "ci_lower", "ci_upper", "n_obs"]].to_string(index=False))
    except Exception as e:
        print(f"  (segment CATE skipped: {e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> CausalPricingModel:
    df = simulate_conversion_data(n=8_000)

    print(f"Conversion data: {len(df):,} quotes")
    print(f"Overall conversion rate: {df['converted'].mean():.1%}")
    print(f"Conversion rate — control: {df.loc[df['discount_flag']==0,'converted'].mean():.1%}")
    print(f"Conversion rate — treated: {df.loc[df['discount_flag']==1,'converted'].mean():.1%}")

    model = fit_causal_model(df)
    uplift_analysis(df, model)
    return model


if __name__ == "__main__":
    main()
