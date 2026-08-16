"""Model comparison: Gini coefficient and actual-vs-expected by decile."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalised_gini(actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> float:
    """
    Exposure-weighted normalised Gini coefficient for a pure-premium prediction.

    Ranks policies by predicted risk, then measures how much of total actual
    cost is concentrated in the highest-predicted-risk exposure — the
    standard way UK pricing teams compare candidate models on lift, not just
    average accuracy. 0 = no better than random ranking, 1 = perfect ranking.
    """
    order = np.argsort(predicted)
    actual, weight = actual[order], weight[order]

    cum_weight = np.cumsum(weight) / weight.sum()
    cum_actual = np.cumsum(actual * weight) / np.sum(actual * weight)

    lorenz_area = np.trapezoid(cum_actual, cum_weight)
    gini = 1 - 2 * lorenz_area
    return float(gini)


def actual_vs_expected_by_decile(
    actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """
    Exposure-weighted actual vs. expected, by decile of predicted risk.

    A well-calibrated model should show ``actual`` tracking ``expected``
    closely within every bin, not just on average across the whole book.
    """
    df = pd.DataFrame({"actual": actual, "predicted": predicted, "weight": weight})
    df["decile"] = pd.qcut(df["predicted"], n_bins, labels=False, duplicates="drop")

    summary = df.groupby("decile", observed=True).apply(
        lambda g: pd.Series({
            "exposure": g["weight"].sum(),
            "actual": np.average(g["actual"], weights=g["weight"]),
            "expected": np.average(g["predicted"], weights=g["weight"]),
        }),
        include_groups=False,
    ).reset_index()
    summary["ave_ratio"] = summary["actual"] / summary["expected"]
    return summary
