import numpy as np
import pytest

from moto_pricing.data.build_dataset import build_modelling_dataset
from moto_pricing.features import LAST_TRAIN_YEAR, out_of_time_split
from moto_pricing.models.ensemble import EnsemblePricingModel
from moto_pricing.models.gbm import GBMPricingModel
from moto_pricing.models.glm import GLMPricingModel


@pytest.fixture(scope="module")
def train_test():
    ds = build_modelling_dataset(15_000, seed=21)
    return out_of_time_split(ds.modelling)


@pytest.mark.parametrize("model_cls", [GLMPricingModel, GBMPricingModel])
def test_model_predictions_are_positive_and_finite(train_test, model_cls):
    train, test = train_test
    model = model_cls().fit(train) if model_cls is GLMPricingModel else model_cls().fit(train, eval_set=test)

    freq = model.predict_frequency(test)
    sev = model.predict_severity(test)
    pp = model.predict_pure_premium(test)

    for arr in (freq, sev, pp):
        assert np.isfinite(arr).all()
        assert (arr > 0).all()
    assert np.allclose(pp, freq * sev)


@pytest.mark.parametrize("model_cls", [GLMPricingModel, GBMPricingModel])
def test_model_recovers_portfolio_mean_frequency(train_test, model_cls):
    """A correctly specified two-part model's predicted mean frequency
    should be within a small margin of the observed portfolio mean."""
    train, test = train_test
    model = model_cls().fit(train) if model_cls is GLMPricingModel else model_cls().fit(train, eval_set=test)

    predicted_mean = model.predict_frequency(test).mean()
    actual_mean = (test["claim_count"] / test["exposure"]).mean()
    assert predicted_mean == pytest.approx(actual_mean, rel=0.25)


def test_ensemble_blend_weights_are_valid_convex_combination(train_test):
    train, _ = train_test
    ensemble = EnsemblePricingModel().fit(train, stacking_val_year=LAST_TRAIN_YEAR)
    assert 0.0 <= ensemble.freq_weight_ <= 1.0
    assert 0.0 <= ensemble.sev_weight_ <= 1.0


def test_ensemble_reuses_provided_base_learners(train_test):
    train, test = train_test
    glm = GLMPricingModel().fit(train)
    gbm = GBMPricingModel().fit(train, eval_set=test)

    ensemble = EnsemblePricingModel().fit(
        train, stacking_val_year=LAST_TRAIN_YEAR, fitted_glm=glm, fitted_gbm=gbm
    )
    assert ensemble.glm is glm
    assert ensemble.gbm is gbm
