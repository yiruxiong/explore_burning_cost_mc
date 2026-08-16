"""
Technical (risk) premium to gross (quoted) premium.

Standard UK personal lines loading structure: fixed per-policy expenses and
variable costs (acquisition, commission, profit) are loaded proportionally
so the insurer clears its target margin, then Insurance Premium Tax is added
on top — IPT is a tax on the premium the customer pays, not a cost the
insurer nets before its margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moto_pricing.config import IPT_RATE


@dataclass(frozen=True)
class PricingAssumptions:
    fixed_expenses: float = 35.0       # per-policy admin cost (£)
    variable_expense_rate: float = 0.09  # acquisition/servicing, % of gross premium
    commission_rate: float = 0.15        # broker commission, % of gross premium
    profit_margin: float = 0.06          # target underwriting margin, % of gross premium
    ipt_rate: float = IPT_RATE

    @property
    def loading_factor(self) -> float:
        retained = 1 - self.variable_expense_rate - self.commission_rate - self.profit_margin
        if retained <= 0:
            raise ValueError("variable_expense_rate + commission_rate + profit_margin must be < 1")
        return 1 / retained


def technical_to_gross(risk_premium: np.ndarray, assumptions: PricingAssumptions | None = None) -> dict[str, np.ndarray]:
    """
    Load an annualised risk premium up to a gross (pre-tax) and payable (with IPT) premium.

    Parameters
    ----------
    risk_premium : np.ndarray
        Annualised pure premium (frequency x severity) per policy.
    assumptions : PricingAssumptions | None
        Expense, commission, profit and tax assumptions. Defaults are used if omitted.

    Returns
    -------
    dict[str, np.ndarray]
        ``risk_premium``, ``gross_premium`` (before IPT) and ``payable_premium``
        (what the customer is quoted, IPT included).
    """
    a = assumptions or PricingAssumptions()
    gross_premium = (risk_premium + a.fixed_expenses) * a.loading_factor
    payable_premium = gross_premium * (1 + a.ipt_rate)
    return {
        "risk_premium": risk_premium,
        "gross_premium": gross_premium,
        "payable_premium": payable_premium,
    }
