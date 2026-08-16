"""
Fairness audit for motorcycle insurance pricing under the UK Equality Act 2010.

Uses :mod:`insurance_fairness` to audit the ensemble pricing model for
indirect discrimination on protected characteristics.  Age is a legitimate
actuarial rating factor; gender is not permitted as a rating factor under
EU/UK Equality Act principles (Test-Achats ruling).

The audit checks:
- Demographic parity (calibration by group)
- Proxy detection — does any rating factor act as a proxy for gender?
- Indirect discrimination test (partial correlation residual method)
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from insurance_fairness import FairnessAudit


# Gender is a protected characteristic — not a rating factor
PROTECTED_COLS = ["gender"]

# Rating factors used in the model
FACTOR_COLS = [
    "age", "years_licensed", "bike_cc", "bike_age",
    "annual_mileage", "ncb", "region", "occupation",
]


def run_fairness_audit(
    data: pd.DataFrame,
    model,
    predictions: np.ndarray,
    outcome_col: str = "total_loss",
) -> FairnessAudit:
    """Run a full fairness audit against the protected characteristic (gender).

    Parameters
    ----------
    data:
        Portfolio data including both rating factors and protected columns.
    model:
        Fitted model object (ensemble model).  Passed to the audit for
        SHAP-based proxy detection.
    predictions:
        Array of model predictions aligned to ``data``.
    outcome_col:
        Column name for the observed outcome (used as the ground truth).

    Returns
    -------
    :class:`~insurance_fairness.FairnessAudit`
    """
    df = data.copy()
    df["prediction"] = predictions

    audit = FairnessAudit(
        model=model,
        data=df,
        protected_cols=PROTECTED_COLS,
        prediction_col="prediction",
        outcome_col=outcome_col,
        exposure_col="exposure",
        factor_cols=FACTOR_COLS,
        model_name="Motorcycle Insurance Ensemble v1",
        run_proxy_detection=True,
        run_counterfactual=False,
        n_bootstrap=0,
        proxy_catboost_iterations=50,
    )
    return audit


def print_fairness_summary(audit: FairnessAudit) -> None:
    """Print a concise fairness audit summary to stdout.

    Parameters
    ----------
    audit:
        Completed :class:`~insurance_fairness.FairnessAudit` (after calling
        ``.run()`` or equivalent).
    """
    print("\n" + "=" * 60)
    print("  FAIRNESS AUDIT SUMMARY — Motorcycle Insurance")
    print("=" * 60)

    report = audit.report_
    if report is None:
        print("  Audit not yet run.  Call audit.run() first.")
        print("=" * 60)
        return

    # Print available metrics
    if hasattr(report, "demographic_parity"):
        dp = report.demographic_parity
        print(f"  Demographic Parity (gender) : {dp}")

    if hasattr(report, "proxy_detection"):
        print(f"  Proxy detection results     : {report.proxy_detection}")

    if hasattr(report, "indirect_discrimination"):
        print(f"  Indirect discrimination     : {report.indirect_discrimination}")

    print("=" * 60 + "\n")
