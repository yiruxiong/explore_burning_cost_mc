"""
Model monitoring and drift detection for motorcycle insurance pricing.

Uses :mod:`insurance_monitoring` to run a full monitoring suite on the
deployed ensemble model, including:

- PSI / CSI feature drift
- A/E ratio monitoring (frequency and severity)
- Gini discrimination test
- Murphy decomposition (calibration vs discrimination drift)
- Conformal SPC chart for premium adequacy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from insurance_monitoring import MonitoringReport, MonitoringThresholds


def run_monitoring(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    reference_predictions: np.ndarray,
    current_predictions: np.ndarray,
    feature_cols: list[str] | None = None,
) -> MonitoringReport:
    """Run a full monitoring check comparing current period to reference.

    Parameters
    ----------
    reference_df:
        Reference period portfolio data (training or first deployment quarter).
    current_df:
        Current monitoring period data.
    reference_predictions:
        Ensemble pure-premium predictions on the reference period.
    current_predictions:
        Ensemble pure-premium predictions on the current period.
    feature_cols:
        List of feature columns for CSI calculation.  Defaults to the
        main continuous rating factors.

    Returns
    -------
    :class:`~insurance_monitoring.MonitoringReport`
    """
    if feature_cols is None:
        feature_cols = [
            "age", "bike_cc", "annual_mileage", "bike_value", "ncb", "bike_age",
        ]

    feature_df_ref = pl.from_pandas(reference_df[feature_cols].astype(float))
    feature_df_cur = pl.from_pandas(current_df[feature_cols].astype(float))

    report = MonitoringReport(
        reference_actual=reference_df["total_loss"].values,
        reference_predicted=reference_predictions,
        current_actual=current_df["total_loss"].values,
        current_predicted=current_predictions,
        exposure=current_df["exposure"].values,
        reference_exposure=reference_df["exposure"].values,
        feature_df_reference=feature_df_ref,
        feature_df_current=feature_df_cur,
        features=feature_cols,
        score_reference=np.log(reference_predictions.clip(1e-6)),
        score_current=np.log(current_predictions.clip(1e-6)),
        thresholds=MonitoringThresholds(),
        n_bootstrap=200,
        murphy_distribution="tweedie",
        gini_bootstrap=False,
    )
    return report


def print_monitoring_summary(report: MonitoringReport) -> None:
    """Print a concise traffic-light monitoring summary to stdout.

    Parameters
    ----------
    report:
        Fitted :class:`~insurance_monitoring.MonitoringReport`.
    """
    print("\n" + "=" * 60)
    print("  MODEL MONITORING SUMMARY — Motorcycle Insurance")
    print("=" * 60)

    results = report.results_

    # A/E ratio
    if "ae_ratio" in results:
        ae = results["ae_ratio"]
        print(f"  A/E Ratio        : {ae.get('value', '?'):.3f}  [{ae.get('band', '?').upper()}]")

    # Gini / discrimination
    if "gini" in results:
        g = results["gini"]
        print(
            f"  Gini (ref/cur)   : {g.get('reference','?'):.3f} → {g.get('current','?'):.3f}  "
            f"[{g.get('band','?').upper()}]"
        )

    # PSI (score drift)
    if "score_psi" in results:
        p = results["score_psi"]
        print(f"  Score PSI        : {p.get('value','?'):.4f}  [{p.get('band','?').upper()}]")

    # Max CSI
    if "max_csi" in results:
        mc = results["max_csi"]
        print(
            f"  Max CSI          : {mc.get('value','?'):.4f}  "
            f"[worst: {mc.get('worst_feature','?')}]  [{mc.get('band','?').upper()}]"
        )

    # Murphy decomposition
    if "murphy" in results:
        m = results["murphy"]
        print(f"  Murphy verdict   : {m.get('verdict','?')}")
        print(f"    Miscalibration : {m.get('miscalibration_pct','?'):.1f}% | "
              f"Discrimination: {m.get('discrimination_pct','?'):.1f}%")

    print("=" * 60 + "\n")
