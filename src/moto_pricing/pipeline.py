"""
End-to-end orchestration: synthetic book -> ensemble pricing models ->
SHAP relativities -> quote-to-bind-to-renewal market simulation -> report.

See ``scripts/run_pipeline.py`` for the CLI entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from moto_pricing.data.build_dataset import ModellingDataset, build_modelling_dataset
from moto_pricing.data.synthetic import generate_policies
from moto_pricing.evaluation import actual_vs_expected_by_decile, normalised_gini
from moto_pricing.features import LAST_TRAIN_YEAR, out_of_time_split
from moto_pricing.journey.conversion import simulate_new_business_conversion
from moto_pricing.journey.lifecycle import simulate_renewals
from moto_pricing.journey.quotes import simulate_quotes
from moto_pricing.models.ensemble import EnsemblePricingModel
from moto_pricing.models.gbm import GBMPricingModel
from moto_pricing.models.glm import GLMPricingModel
from moto_pricing.relativities import extract_frequency_relativities
from moto_pricing.reporting import (
    plot_actual_vs_expected,
    plot_categorical_relativities,
    plot_model_comparison,
    plot_quote_funnel,
)

MODELS = {"glm": GLMPricingModel, "gbm": GBMPricingModel, "ensemble": None}  # ensemble built separately


@dataclass
class PipelineResult:
    dataset: ModellingDataset
    models: dict[str, object]
    model_comparison: pd.DataFrame
    frequency_relativities: pd.DataFrame
    quotes: pd.DataFrame
    renewals: pd.DataFrame
    funnel: pd.DataFrame


def _evaluate(model, test: pd.DataFrame) -> dict:
    pred_pp = model.predict_pure_premium(test)
    actual_pp = test["incurred"].to_numpy() / np.clip(test["exposure"].to_numpy(), 1e-6, None)
    gini = normalised_gini(actual_pp, pred_pp, test["exposure"].to_numpy())
    ave = actual_vs_expected_by_decile(actual_pp, pred_pp, test["exposure"].to_numpy())
    return {"gini": gini, "ave": ave}


def run_full_pipeline(
    n_policies: int = 150_000,
    n_prospects: int = 40_000,
    seed: int = 42,
    outputs_dir: Path | None = None,
) -> PipelineResult:
    """
    Run the full lifecycle: build data, train/compare models, extract
    relativities, simulate the quote-to-bind-to-renewal journey, and (if
    ``outputs_dir`` is given) write a report to disk.

    Parameters
    ----------
    n_policies : int
        Size of the historical book used to train and evaluate the pricing models.
    n_prospects : int
        Size of the new-business quote cohort priced by the fitted model and
        run through the conversion/renewal simulation.
    seed : int
        Base random seed; every stage derives its own seed from this one.
    outputs_dir : Path | None
        If given, CSV tables and PNG charts are written here.

    Returns
    -------
    PipelineResult
    """
    dataset = build_modelling_dataset(n_policies, seed=seed)
    train, test = out_of_time_split(dataset.modelling)

    glm = GLMPricingModel().fit(train)
    gbm = GBMPricingModel().fit(train, eval_set=test)
    ensemble = EnsemblePricingModel().fit(
        train, stacking_val_year=LAST_TRAIN_YEAR, fitted_glm=glm, fitted_gbm=gbm
    )
    models = {"glm": glm, "gbm": gbm, "ensemble": ensemble}

    comparison_rows = []
    ave_tables = {}
    for name, model in models.items():
        metrics = _evaluate(model, test)
        comparison_rows.append({"model": name, "gini": metrics["gini"]})
        ave_tables[name] = metrics["ave"]
    model_comparison = pd.DataFrame(comparison_rows)

    frequency_relativities = extract_frequency_relativities(
        gbm.frequency_model, train, train["exposure"]
    )

    prospects = generate_policies(n_prospects, seed=seed + 1000)
    prospects["exposure"] = 1.0
    quotes = simulate_quotes(prospects, ensemble, seed=seed + 1001)
    converted = simulate_new_business_conversion(quotes, seed=seed + 1002)
    bound = converted[converted["bound"]].reset_index(drop=True)
    renewals = simulate_renewals(bound, ensemble, claims_seed=seed + 1003, retention_seed=seed + 1004)

    funnel = pd.DataFrame({
        "stage": ["Quoted", "Bound", "Renewed"],
        "count": [len(quotes), len(bound), int(renewals["renewed"].sum())],
    })

    if outputs_dir is not None:
        _write_report(
            outputs_dir, dataset, model_comparison, ave_tables,
            frequency_relativities, quotes, converted, renewals, funnel,
        )

    return PipelineResult(
        dataset=dataset, models=models, model_comparison=model_comparison,
        frequency_relativities=frequency_relativities, quotes=converted,
        renewals=renewals, funnel=funnel,
    )


def _write_report(
    outputs_dir: Path,
    dataset: ModellingDataset,
    model_comparison: pd.DataFrame,
    ave_tables: dict[str, pd.DataFrame],
    frequency_relativities: pd.DataFrame,
    quotes: pd.DataFrame,
    converted: pd.DataFrame,
    renewals: pd.DataFrame,
    funnel: pd.DataFrame,
) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = outputs_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    model_comparison.to_csv(outputs_dir / "model_comparison.csv", index=False)

    categorical = frequency_relativities[
        frequency_relativities["level"].apply(lambda v: isinstance(v, str))
    ]
    categorical.to_csv(outputs_dir / "frequency_relativities.csv", index=False)

    dataset.exposure_by_accident_year.to_csv(outputs_dir / "exposure_by_accident_year.csv")

    conversion_summary = (
        converted.groupby(pd.cut(converted["price_gap"], 10))["bound"]
        .mean()
        .rename("conversion_rate")
        .reset_index()
    )
    conversion_summary.to_csv(outputs_dir / "conversion_by_price_gap.csv", index=False)

    renewals[[
        "policy_id", "prior_payable_premium", "renewal_payable_premium",
        "renewal_price_change", "year_1_claim_count", "retention_probability", "renewed",
    ]].to_csv(outputs_dir / "renewals.csv", index=False)

    funnel.to_csv(outputs_dir / "funnel.csv", index=False)

    plot_model_comparison(model_comparison, plots_dir / "model_comparison.png")
    plot_actual_vs_expected(ave_tables["ensemble"], "ensemble", plots_dir / "actual_vs_expected_ensemble.png")
    for feature in ("bike_type", "area", "security"):
        plot_categorical_relativities(categorical, feature, plots_dir / f"relativities_{feature}.png")
    plot_quote_funnel(funnel, plots_dir / "quote_funnel.png")

    summary_lines = [
        "# Motorcycle Pricing Pipeline — Run Summary",
        "",
        f"Historical book: {len(dataset.policies):,} policies, {len(dataset.claims):,} claims.",
        "",
        "## Model comparison (out-of-time Gini, pure premium)",
        model_comparison.to_markdown(index=False),
        "",
        "## Quote-to-bind-to-renewal funnel",
        funnel.to_markdown(index=False),
        "",
        f"New-business conversion rate: {converted['bound'].mean():.1%}",
        f"Renewal retention rate: {renewals['renewed'].mean():.1%}",
        f"Mean renewal price change: {renewals['renewal_price_change'].mean():+.1%}",
    ]
    (outputs_dir / "pipeline_report.md").write_text("\n".join(summary_lines) + "\n")
