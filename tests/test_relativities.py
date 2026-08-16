import pytest

from moto_pricing.data.build_dataset import build_modelling_dataset
from moto_pricing.models.gbm import GBMPricingModel
from moto_pricing.relativities import extract_frequency_relativities


@pytest.fixture(scope="module")
def frequency_relativities():
    ds = build_modelling_dataset(15_000, seed=31)
    gbm = GBMPricingModel().fit(ds.modelling)
    return extract_frequency_relativities(gbm.frequency_model, ds.modelling, ds.modelling["exposure"])


def test_base_levels_have_relativity_one(frequency_relativities):
    base_level_rows = frequency_relativities[
        (frequency_relativities["feature"] == "bike_type") & (frequency_relativities["level"] == "Tourer")
    ]
    assert base_level_rows["relativity"].iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_supersport_relativity_exceeds_tourer(frequency_relativities):
    bike_type = frequency_relativities[frequency_relativities["feature"] == "bike_type"]
    supersport = bike_type.loc[bike_type["level"] == "Supersport", "relativity"].iloc[0]
    tourer = bike_type.loc[bike_type["level"] == "Tourer", "relativity"].iloc[0]
    assert supersport > tourer


def test_area_f_relativity_exceeds_area_a(frequency_relativities):
    area = frequency_relativities[frequency_relativities["feature"] == "area"]
    area_f = area.loc[area["level"] == "F", "relativity"].iloc[0]
    area_a = area.loc[area["level"] == "A", "relativity"].iloc[0]
    assert area_f > area_a
