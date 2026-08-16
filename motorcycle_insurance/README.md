# Motorcycle Insurance Pricing — Full Lifecycle with `burning-cost`

A complete UK personal lines **motorcycle insurance pricing** project built on
the [`burning-cost`](https://github.com/burningcost/burning-cost) ecosystem.

Covers the **full model lifecycle**: synthetic data → ensemble model fitting →
quote-to-pricing → pricing-to-sell → monitoring → fairness audit →
conformal prediction → governance documentation.

---

## Project Structure

```
motorcycle_insurance/
├── data/
│   └── synthetic.py          # Synthetic UK motorcycle portfolio generator (50 000 policies)
├── models/
│   ├── frequency_severity.py # Poisson + Gamma GLM baseline (statsmodels)
│   ├── gam_model.py          # Penalized GLM / GAM (insurance-gam Elastic Net)
│   ├── credibility.py        # Bühlmann-Straub + Poisson-Gamma credibility weighting
│   └── ensemble.py           # Ensemble pipeline: GLM + GAM + Credibility
├── pipelines/
│   ├── quote_to_price.py     # Quote-to-pricing journey (loading, referrals)
│   └── price_to_sell.py      # Price-to-sell simulation (elasticity, conversion)
├── monitoring/
│   └── drift.py              # Model monitoring: A/E, Gini, PSI/CSI, Murphy
├── fairness/
│   └── audit.py              # Fairness audit under UK Equality Act 2010
├── conformal/
│   └── intervals.py          # Conformal prediction intervals (90% coverage)
├── governance/
│   └── audit.py              # Model card + governance report (FCA/PRA aligned)
└── main.py                   # End-to-end orchestration script
```

---

## Libraries Used

| Library | Role |
|---|---|
| `insurance-frequency-severity` | Poisson/Gamma frequency-severity GLMs |
| `insurance-gam` | Penalized GLM (Elastic Net) for smooth continuous effects |
| `insurance-credibility` | Bühlmann-Straub and Poisson-Gamma credibility weighting |
| `insurance-monitoring` | A/E ratio, Gini drift, PSI/CSI, Murphy decomposition |
| `insurance-conformal` | Distribution-free prediction intervals |
| `insurance-fairness` | Demographic parity, proxy detection, indirect discrimination |
| `insurance-governance` | MRM model card, risk tier scoring, governance report |

---

## Quick Start

```bash
# Install the burning-cost meta-package (installs all 10 libraries)
pip install burning-cost

# Run the full lifecycle from the repo root
cd explore_burning_cost_mc
PYTHONPATH=. python motorcycle_insurance/main.py
```

---

## Full Lifecycle Steps

### 1 · Synthetic Data Generation

`data/synthetic.py` generates 50 000 synthetic UK motorcycle insurance policies
(2018–2023) with realistic distributions for:

- Rider age (U-shaped risk: young and older riders)
- Engine displacement (125cc → 1300cc)
- Annual mileage, bike value, NCB steps
- 11 UK regions with realistic area relativities
- Occupation categories with risk loadings
- Poisson claim counts with Gamma severity
- Price-sensitivity conversion curve (sigmoid elasticity)

### 2 · Ensemble Model

The ensemble combines three complementary approaches:

1. **GLM baseline** — Poisson frequency + Gamma severity with categorical
   region and occupation effects.  Full interpretability via statsmodels
   summary tables.

2. **Penalized GLM (GAM-style)** — `insurance-gam` Elastic Net regularisation
   smooths continuous rating factors (age, mileage, engine size).  Provides
   bias-corrected confidence intervals via `PenalizedGLMInference`.

3. **Credibility weighting** — Bühlmann-Straub blends region A/E ratios;
   Poisson-Gamma conjugate blends occupation frequency rates.  Protects
   against overfitting in thin-data segments.

Ensemble blend weights are optimised by minimising Poisson deviance on a
held-out validation year.

### 3 · Quote-to-Pricing Journey

`pipelines/quote_to_price.py` simulates the full quote journey:

- Technical pure premium from the ensemble model
- Gross premium loading (expenses, profit margin, reinsurance)
- Competitive market noise (±5%)
- Referral rules (frequency threshold, premium ceiling)

### 4 · Pricing-to-Sell Journey

`pipelines/price_to_sell.py` models customer purchase decisions:

- Competitive market price simulation (log-normal noise)
- Logistic conversion curve with configurable price elasticity
- Renewal vs. new business elasticity
- Portfolio economics summary (conversion rate, written premium, loss ratio)

### 5 · Model Monitoring

`monitoring/drift.py` uses `insurance-monitoring` to run quarterly checks:

| Metric | Description |
|---|---|
| **A/E ratio** | Actual vs. expected claims frequency |
| **Gini drift** | Discrimination deterioration test |
| **Score PSI** | Log-rate score population stability index |
| **CSI** | Characteristic stability index per feature |
| **Murphy decomposition** | Distinguishes calibration drift vs. discrimination drift → `RECALIBRATE` or `REFIT` |

### 6 · Conformal Prediction Intervals

`conformal/intervals.py` wraps the ensemble in an
`InsuranceConformalPredictor` to produce **distribution-free 90% prediction
intervals** for individual policy pure premiums.  The calibration split follows
temporal ordering: train on 2018–2021, calibrate on 2022, predict on 2023.

### 7 · Fairness Audit

`fairness/audit.py` runs an `insurance-fairness` audit on **gender** as a
protected characteristic under the UK Equality Act 2010:

- Demographic parity ratio (calibration by gender group)
- Proxy detection — checks whether any rating factor acts as a statistical
  proxy for gender
- Indirect discrimination test (partial correlation residual method)

### 8 · Governance Report

`governance/audit.py` produces:

- **MRM Model Card** (`MRMModelCard`) capturing model metadata required by
  FCA/PRA model risk management frameworks
- **Risk Tier Score** (`RiskTierScorer`) to classify model materiality
- **Governance Report** (`GovernanceReport`) HTML/JSON artefact for the
  Model Risk Committee

---

## Sample Output

```
[1/9] Generating synthetic UK motorcycle insurance portfolio...
      Portfolio: 50,000 policies | claim rate: 0.049 | avg severity: £1,132

[3/9] Fitting ensemble pricing model...
  Ensemble weights: [0.333 0.333 0.333]
  Test A/E ratio: 1.03 | Approx Gini: 0.162

[6/9] Running model monitoring...
  A/E Ratio        : 1.03   [GREEN]
  Gini (ref/cur)   : 0.235 → 0.160  [AMBER]
  Score PSI        : 0.004  [GREEN]
  Murphy verdict   : REFIT

[7/9] Conformal intervals (first 5 policies):
  lower    point    upper
    0.0   37.78   105.26
    0.0   42.40   115.98
```

---

## Design Principles

- **Clean and concise** — each module has a single responsibility with
  minimal boilerplate
- **Readable** — docstrings explain both the actuarial rationale and the
  API contract
- **Realistic** — distributions, loadings, and elasticity parameters
  reflect UK personal lines motorcycle market norms
- **Full lifecycle** — covers every stage from raw data through to MRM
  governance documentation
