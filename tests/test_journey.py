import numpy as np

from moto_pricing.data.synthetic import generate_policies
from moto_pricing.journey.conversion import simulate_new_business_conversion
from moto_pricing.journey.lifecycle import MAX_NCD_YEARS, simulate_renewals
from moto_pricing.journey.quotes import simulate_quotes


class _ConstantRiskModel:
    """Prices every policy at a fixed risk premium — fast stand-in for a
    fitted GLM/GBM/ensemble in tests that only exercise the journey glue."""

    def predict_pure_premium(self, df):
        return np.full(len(df), 150.0)


def _prospects(n=2_000, seed=1):
    df = generate_policies(n, seed=seed)
    df["exposure"] = 1.0
    return df


def test_simulate_quotes_produces_a_price_gap():
    quotes = simulate_quotes(_prospects(), _ConstantRiskModel(), seed=2)
    assert {"payable_premium", "market_price", "price_gap"} <= set(quotes.columns)
    assert (quotes["payable_premium"] > 0).all()
    assert (quotes["market_price"] > 0).all()


def test_cheaper_quotes_convert_more_often():
    quotes = simulate_quotes(_prospects(5_000), _ConstantRiskModel(), seed=3)
    converted = simulate_new_business_conversion(quotes, seed=4)

    cheap = converted[converted["price_gap"] < converted["price_gap"].median()]
    expensive = converted[converted["price_gap"] >= converted["price_gap"].median()]
    assert cheap["bound"].mean() > expensive["bound"].mean()


def test_simulate_renewals_bounds_ncd_and_probabilities():
    quotes = simulate_quotes(_prospects(3_000), _ConstantRiskModel(), seed=5)
    converted = simulate_new_business_conversion(quotes, seed=6)
    bound = converted[converted["bound"]].reset_index(drop=True)

    renewals = simulate_renewals(bound, _ConstantRiskModel(), claims_seed=7, retention_seed=8)

    assert renewals["ncd_years"].between(0, MAX_NCD_YEARS).all()
    assert renewals["retention_probability"].between(0, 1).all()
    assert renewals["renewal_payable_premium"].gt(0).all()


def test_a_claim_reduces_ncd_relative_to_a_claim_free_renewal():
    quotes = simulate_quotes(_prospects(4_000), _ConstantRiskModel(), seed=9)
    converted = simulate_new_business_conversion(quotes, seed=10)
    bound = converted[converted["bound"]].reset_index(drop=True)
    bound["ncd_years"] = 5  # fix starting NCD so the comparison isn't confounded

    renewals = simulate_renewals(bound, _ConstantRiskModel(), claims_seed=11, retention_seed=12)

    claim_free = renewals[renewals["year_1_claim_count"] == 0]
    had_claim = renewals[renewals["year_1_claim_count"] > 0]
    assert claim_free["ncd_years"].mean() > had_claim["ncd_years"].mean()
