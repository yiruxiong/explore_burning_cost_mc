"""
Model governance documentation for motorcycle insurance pricing.

Uses :mod:`insurance_governance` to produce:

- A structured :class:`~insurance_governance.MRMModelCard` capturing all
  model metadata required by FCA/PRA model risk management frameworks.
- A :class:`~insurance_governance.RiskTierScorer` assessment to classify
  model materiality (Tier 1 / 2 / 3).
- A :class:`~insurance_governance.GovernanceReport` HTML/JSON artefact
  for the Model Risk Committee.
"""

from __future__ import annotations

from insurance_governance import MRMModelCard, GovernanceReport, RiskTierScorer


RATING_FACTORS = [
    "age",
    "years_licensed",
    "bike_cc",
    "bike_age",
    "bike_value",
    "annual_mileage",
    "ncb",
    "region",
    "occupation",
]


def build_model_card() -> MRMModelCard:
    """Build the MRM model card for the motorcycle ensemble pricing model.

    Returns
    -------
    :class:`~insurance_governance.MRMModelCard`
    """
    card = MRMModelCard(
        model_id="mc-ensemble-freq-sev-v1",
        model_name="UK Motorcycle Insurance Ensemble Pricing Model",
        version="1.0.0",
        model_class="pricing",
        intended_use=(
            "Compute risk-based technical premiums for UK private motorcycle "
            "insurance policies at the point of quote. Output is used as the "
            "technical rate input to the pricing ladder."
        ),
        not_intended_for=[
            "Commercial vehicle or fleet motorcycle pricing",
            "Solvency II internal model capital calculations",
            "Credit scoring or eligibility decisions",
        ],
        target_variable="pure_premium",
        distribution_family="Tweedie",
        model_type="Ensemble: Poisson/Gamma GLM + Penalized GLM (Elastic Net) + Bühlmann-Straub Credibility",
        rating_factors=RATING_FACTORS,
        training_data_period=("2018-01-01", "2022-12-31"),
        development_date="2023-06-01",
        developer="Actuarial Pricing Team",
        champion_challenger_status="development",
        portfolio_scope="UK private motorcycle insurance, comprehensive and third-party policies",
        geographic_scope="United Kingdom (England, Scotland, Wales)",
        customer_facing=True,
        regulatory_use=False,
        gwp_impacted=15_000_000.0,
        monitoring_owner="Chief Actuary",
        monitoring_frequency="Quarterly",
        monitoring_triggers={
            "ae_ratio_deviation": 0.10,
            "gini_drop": 0.05,
            "score_psi": 0.25,
        },
        trigger_actions={
            "ae_ratio_deviation": "Investigate and recalibrate if systematic",
            "gini_drop": "Trigger model refit review",
            "score_psi": "Investigate population shift; consider recalibration",
        },
    )
    return card


def assess_model_tier(card: MRMModelCard) -> dict:
    """Compute the model risk tier using :class:`~insurance_governance.RiskTierScorer`.

    Parameters
    ----------
    card:
        Populated :class:`~insurance_governance.MRMModelCard`.

    Returns
    -------
    dict with ``tier_result`` and ``tier`` (int).
    """
    scorer = RiskTierScorer()
    tier_result = scorer.score(
        gwp_impacted=card.gwp_impacted,
        model_complexity="medium",
        deployment_status=card.champion_challenger_status,
        regulatory_use=card.regulatory_use,
        external_data=False,
        customer_facing=card.customer_facing,
        validation_months_ago=None,  # not yet validated
        drift_triggers_last_year=0,
    )
    return {"tier_result": tier_result, "tier": tier_result.tier}


def build_governance_report(
    card: MRMModelCard,
    monitoring_results: dict | None = None,
    validation_results: dict | None = None,
) -> GovernanceReport:
    """Build a full governance report artefact.

    Parameters
    ----------
    card:
        Populated model card.
    monitoring_results:
        Optional monitoring output dict (from MonitoringReport).
    validation_results:
        Optional validation output dict.

    Returns
    -------
    :class:`~insurance_governance.GovernanceReport`
    """
    tier_info = assess_model_tier(card)
    report = GovernanceReport(
        card=card,
        tier=tier_info["tier_result"],
        validation_results=validation_results or {},
        monitoring_results=monitoring_results or {},
    )
    return report
