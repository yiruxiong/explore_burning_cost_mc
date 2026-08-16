import numpy as np

from moto_pricing.evaluation import actual_vs_expected_by_decile, normalised_gini


def test_gini_ranks_a_perfect_model_far_above_an_uninformative_one():
    """A normalised Gini caps out at the underlying loss distribution's own
    concentration (e.g. exactly 0.5 for an Exponential distribution, a known
    result) — it never reaches 1 for continuous individual-level losses. So
    the meaningful check is perfect ranking vs. no ranking, not an absolute
    threshold."""
    rng = np.random.default_rng(0)
    actual = rng.exponential(size=5_000)
    weight = np.ones_like(actual)

    perfect = normalised_gini(actual, predicted=actual, weight=weight)
    uninformative = normalised_gini(actual, predicted=np.ones_like(actual), weight=weight)

    assert perfect > uninformative + 0.3


def test_gini_is_near_zero_for_an_uninformative_prediction():
    rng = np.random.default_rng(0)
    actual = rng.exponential(size=5_000)
    predicted = np.ones_like(actual)  # every policy priced identically
    weight = np.ones_like(actual)
    gini = normalised_gini(actual, predicted, weight)
    assert abs(gini) < 0.05


def test_actual_vs_expected_by_decile_shape():
    rng = np.random.default_rng(1)
    n = 2_000
    predicted = rng.uniform(50, 500, n)
    actual = predicted * rng.uniform(0.8, 1.2, n)
    weight = rng.uniform(0.5, 1.0, n)

    result = actual_vs_expected_by_decile(actual, predicted, weight, n_bins=5)

    assert len(result) == 5
    assert {"decile", "exposure", "actual", "expected", "ave_ratio"} <= set(result.columns)
    assert result["ave_ratio"].between(0.5, 1.5).all()
