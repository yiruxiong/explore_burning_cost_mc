import numpy as np
import pytest

from moto_pricing.pricing import PricingAssumptions, technical_to_gross


def test_technical_to_gross_known_values():
    assumptions = PricingAssumptions(
        fixed_expenses=20.0, variable_expense_rate=0.10,
        commission_rate=0.15, profit_margin=0.05, ipt_rate=0.12,
    )
    risk_premium = np.array([100.0])

    result = technical_to_gross(risk_premium, assumptions)

    expected_gross = (100.0 + 20.0) / (1 - 0.10 - 0.15 - 0.05)
    assert result["gross_premium"][0] == pytest.approx(expected_gross)
    assert result["payable_premium"][0] == pytest.approx(expected_gross * 1.12)


def test_technical_to_gross_rejects_impossible_margin():
    assumptions = PricingAssumptions(variable_expense_rate=0.5, commission_rate=0.4, profit_margin=0.2)
    with pytest.raises(ValueError):
        technical_to_gross(np.array([100.0]), assumptions)


def test_higher_risk_premium_always_gives_higher_payable_premium():
    risk_premium = np.array([50.0, 500.0])
    result = technical_to_gross(risk_premium)
    assert result["payable_premium"][1] > result["payable_premium"][0]
