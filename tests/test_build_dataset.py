import pytest

from moto_pricing.data.build_dataset import build_modelling_dataset


def test_build_modelling_dataset_shapes_and_consistency():
    ds = build_modelling_dataset(3_000, seed=11)

    assert len(ds.modelling) == len(ds.policies)
    assert ds.modelling["policy_id"].is_unique
    assert (ds.modelling["exposure"] >= 0).all()
    # Written exposure is actual term days / 365.25, so a 12-month term
    # spanning a leap-year February lands just over 1.0 (366 / 365.25).
    assert (ds.modelling["exposure"] <= 1.01).all()

    # Every claim's cost must have landed on its policy.
    assert ds.modelling["incurred"].sum() == pytest.approx(ds.claims["incurred"].sum())
    assert ds.modelling["claim_count"].sum() == len(ds.claims)

    # Policies with no claims must show zero, not NaN, cost.
    no_claims = ds.modelling[ds.modelling["claim_count"] == 0]
    assert (no_claims["incurred"] == 0.0).all()
    assert no_claims["avg_severity"].isna().all()


def test_exposure_by_accident_year_spans_full_book_window():
    ds = build_modelling_dataset(3_000, seed=12)
    years = ds.exposure_by_accident_year.index
    assert years.min() == 2019
    assert years.max() >= 2023
    assert (ds.exposure_by_accident_year > 0).all()
