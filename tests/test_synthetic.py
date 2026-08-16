import pandas as pd
import pytest

from moto_pricing.data.synthetic import (
    AREA_BANDS,
    BIKE_TYPES,
    generate_claims,
    generate_policies,
    true_expected_pure_premium,
)


def test_generate_policies_shape_and_ranges():
    df = generate_policies(2_000, seed=1)
    assert len(df) == 2_000
    assert df["policy_id"].is_unique
    assert set(df["bike_type"]).issubset(set(BIKE_TYPES))
    assert set(df["area"]).issubset(set(AREA_BANDS))
    assert df["rider_age"].between(17, 85).all()
    assert df["engine_cc"].between(50, 2000).all()
    assert (df["expiry_date"] > df["inception_date"]).all()


def test_generate_policies_is_reproducible():
    a = generate_policies(500, seed=7)
    b = generate_policies(500, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_generate_claims_references_valid_policies():
    policies = generate_policies(5_000, seed=2)
    claims = generate_claims(policies, seed=3)
    assert claims["policy_id"].isin(policies["policy_id"]).all()
    assert (claims["incurred"] > 0).all()
    assert set(claims["peril"]) <= {"accident", "theft", "fire", "third_party_injury"}


def test_security_and_parking_reduce_theft_share():
    """Bikes with the best theft protection should show a materially lower
    theft share among their claims than bikes with none."""
    policies = generate_policies(60_000, seed=4)
    claims = generate_claims(policies, seed=5)
    claims = claims.merge(
        policies[["policy_id", "security", "overnight_parking"]], on="policy_id"
    )

    best = claims[(claims["security"] == "alarm_and_tracker") & (claims["overnight_parking"] == "garage")]
    worst = claims[(claims["security"] == "none") & (claims["overnight_parking"] == "street")]

    theft_share_best = (best["peril"] == "theft").mean()
    theft_share_worst = (worst["peril"] == "theft").mean()
    assert theft_share_best < theft_share_worst


def test_true_expected_pure_premium_matches_realised_average():
    """The DGP's closed-form expectation should track the simulated average
    on a large enough book (law of large numbers, not an exact match)."""
    policies = generate_policies(80_000, seed=8)
    claims = generate_claims(policies, seed=9)

    realised = claims["incurred"].sum() / len(policies)
    expected = true_expected_pure_premium(policies).mean()

    assert realised == pytest.approx(expected, rel=0.15)


def test_young_rider_supersport_riskier_than_mature_tourer():
    policies = generate_policies(10, seed=1).iloc[:2].copy()
    policies.loc[policies.index[0], ["rider_age", "bike_type", "engine_cc", "ncd_years"]] = [
        20, "Supersport", 1000, 0,
    ]
    policies.loc[policies.index[1], ["rider_age", "bike_type", "engine_cc", "ncd_years"]] = [
        45, "Tourer", 900, 5,
    ]
    expected = true_expected_pure_premium(policies)
    assert expected[0] > expected[1]
