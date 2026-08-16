"""
05_quote_to_price.py
====================
Simulate the quote-to-technical-price journey for a stream of 5,000 new
motorcycle quotes in the UK personal lines market.

Pipeline
--------
1. Generate a quote stream of unseen risks (00_data_generation.generate_quote_stream).
2. Apply the ensemble model to produce a technical pure premium.
3. Apply a commercial loading schedule:
     - expense loading   : +12%
     - profit target     : +8%
     - market softening  : –5% (competitive adjustment for online channel)
4. Apply NCD discount schedule on top.
5. Apply regional market factors (London/SE uplift).
6. Produce a final quoted premium with 90% conformal prediction interval.

The output shows, for each quote:
  pure_premium | loaded_premium | quoted_premium | interval_lower | interval_upper
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from mc_pricing.data_generation import generate_full_portfolio, generate_quote_stream


# ---------------------------------------------------------------------------
# Commercial loading schedule
# ---------------------------------------------------------------------------

EXPENSE_LOAD    = 0.12   # +12%
PROFIT_TARGET   = 0.08   # +8%
CHANNEL_SOFT    = 0.05   # –5% online channel discount
IPT_RATE        = 0.12   # UK Insurance Premium Tax

# NCD commercial discount (on top of technical adjustment already in the model)
NCD_COMMERCIAL_DISCOUNT = {0: 0.00, 1: 0.05, 2: 0.10, 3: 0.15, 4: 0.20}

# Regional market adjustment (reflects competitive pressure, not just risk)
REGION_MARKET_ADJ = {
    "London":     1.10,  # high demand, market charges more
    "South East": 1.05,
    "Midlands":   1.00,
    "North":      0.97,
    "Scotland":   0.95,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GLM_FEATURES = [
    "rider_age", "ncd_years", "engine_cc", "vehicle_age",
    "region_Midlands", "region_North", "region_Scotland", "region_South East",
]


def _encode_region(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=float)
    for col in ["region_Midlands", "region_North", "region_Scotland", "region_South East"]:
        if col not in df.columns:
            df[col] = 0.0
    return df


def predict_pure_premium_glm(
    quotes: pd.DataFrame,
    freq_glm: sm.GLM,
    sev_glm: sm.GLM,
) -> np.ndarray:
    X = sm.add_constant(quotes[GLM_FEATURES].astype(float), has_constant="add")
    mu_freq = freq_glm.predict(X)   # exposure assumed 1.0 for a new annual quote
    mu_sev  = sev_glm.predict(X)
    return mu_freq * mu_sev


def apply_commercial_loadings(
    pure_premium: np.ndarray,
    ncd_years: np.ndarray,
    region: pd.Series,
) -> pd.DataFrame:
    # Step 1: gross-up for expenses and profit
    gross_factor = (1 + EXPENSE_LOAD + PROFIT_TARGET) * (1 - CHANNEL_SOFT)
    loaded = pure_premium * gross_factor

    # Step 2: NCD commercial discount
    ncd_disc = np.array([NCD_COMMERCIAL_DISCOUNT[n] for n in ncd_years])
    after_ncd = loaded * (1 - ncd_disc)

    # Step 3: regional market adjustment
    reg_adj = region.map(REGION_MARKET_ADJ).fillna(1.0).values
    after_region = after_ncd * reg_adj

    # Step 4: add IPT (Insurance Premium Tax — not included in pure premium)
    gross_premium = after_region * (1 + IPT_RATE)

    return pd.DataFrame({
        "pure_premium":    np.round(pure_premium, 2),
        "loaded_premium":  np.round(loaded, 2),
        "after_ncd":       np.round(after_ncd, 2),
        "gross_premium":   np.round(gross_premium, 2),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def simulate_quote_to_price(
    freq_glm: sm.GLM,
    sev_glm: sm.GLM,
    conformal=None,
) -> pd.DataFrame:
    quotes_raw = generate_quote_stream(n_quotes=5_000)
    quotes = quotes_raw.copy()
    quotes["exposure"] = 1.0   # annual quote

    # Keep region string before encoding
    region_str = quotes["region"].copy()

    quotes_enc = _encode_region(quotes.copy())

    pp = predict_pure_premium_glm(quotes_enc, freq_glm, sev_glm)

    pricing = apply_commercial_loadings(pp, quotes["ncd_years"].values, region_str)

    result = pd.concat([
        quotes_raw[["quote_id", "rider_age", "ncd_years", "engine_cc",
                     "vehicle_age", "region"]].reset_index(drop=True),
        pricing.reset_index(drop=True),
    ], axis=1)

    # Conformal intervals if available
    if conformal is not None:
        intervals = conformal.predict_interval(quotes_enc, alpha=0.10).to_pandas()
        result["interval_lower"] = np.round(intervals["lower"].values, 2)
        result["interval_upper"] = np.round(intervals["upper"].values, 2)

    print("\n=== Quote-to-Price Sample (first 10) ===")
    cols = ["quote_id", "rider_age", "ncd_years", "engine_cc", "region",
            "pure_premium", "loaded_premium", "gross_premium"]
    if conformal is not None:
        cols += ["interval_lower", "interval_upper"]
    print(result[cols].head(10).to_string(index=False))

    print(f"\nPure premium — mean: £{pp.mean():,.0f}  "
          f"p25: £{np.percentile(pp,25):,.0f}  "
          f"p75: £{np.percentile(pp,75):,.0f}")
    print(f"Gross premium — mean: £{result['gross_premium'].mean():,.0f}")

    return result


if __name__ == "__main__":
    # Stand-alone: fit minimal GLM models for demo
    df_raw = generate_full_portfolio()
    df = pd.get_dummies(df_raw.copy(), columns=["region"], drop_first=True, dtype=float)
    for col in GLM_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df["region"] = df_raw["region"].values

    train = df[df["calendar_year"].isin([2021, 2022])].copy()
    X_tr  = sm.add_constant(train[GLM_FEATURES].astype(float))
    freq_glm = sm.GLM(train["claim_count"], X_tr, family=sm.families.Poisson(),
                      exposure=train["exposure"].values).fit(disp=False)
    claims = train[train["claim_count"] > 0]
    X_cl  = sm.add_constant(claims[GLM_FEATURES].astype(float))
    sev_glm = sm.GLM(claims["avg_severity"], X_cl,
                     family=sm.families.Gamma(link=sm.families.links.Log())).fit(disp=False)

    simulate_quote_to_price(freq_glm, sev_glm)
