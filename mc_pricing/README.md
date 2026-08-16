# mc_pricing — UK Motorcycle Insurance Pricing Models

A complete pricing lifecycle built on the [burning-cost](https://github.com/burningcost/burning-cost)
ecosystem, simulating the full journey from raw data to quoted premium in the UK personal lines
motorcycle market.

---

## What this project does

| Script | Stage | Key library |
|--------|-------|-------------|
| `00_data_generation.py` | Synthetic data | — |
| `01_frequency_severity.py` | GLM freq × sev + Sarmanov copula | `insurance-frequency-severity` |
| `02_gam_model.py` | EBM GAM with relativity tables | `insurance-gam` |
| `03_ensemble_pipeline.py` | GLM + EBM ensemble + conformal intervals | `insurance-conformal` |
| `04_credibility.py` | Bühlmann-Straub regional credibility | `insurance-credibility` |
| `05_quote_to_price.py` | Quote → technical price → gross premium | — |
| `06_price_to_sell.py` | A/B discount test → causal conversion uplift | `insurance-causal` |
| `07_monitoring.py` | Gini drift, A/E, PSI feature monitoring | `insurance-monitoring` |
| `08_fairness_governance.py` | Equality Act fairness audit + MRM model card | `insurance-fairness`, `insurance-governance` |

---

## Setup

```bash
pip install burning-cost interpret
```

---

## Synthetic data (`00_data_generation.py`)

Generates 30,000 policy-years across three calendar years (2021–2023) with:

- **Rider age** — U-shaped frequency curve (young and older riders riskier)
- **NCD years** — discount tiers 0–4 with Sarmanov-style claim-suppression effect
- **Engine cc** — log-linear frequency and severity increases
- **Vehicle age** — older bikes depreciate, lowering severity
- **Region** — 5 UK regions with London/South East uplift
- **Partial exposure** — random mid-term cancellations (0.3–1.0 years)

A shared `data_generation.py` module is importable by all other scripts.

---

## Frequency-Severity GLMs (`01_frequency_severity.py`)

1. Poisson frequency GLM with log-exposure offset (2021–2022 training)
2. Gamma severity GLM on claim rows only
3. `DependenceTest` — Kendall tau + Spearman rho test for NCD-driven dependence
4. `JointFreqSev` Sarmanov copula — per-policy correction factors
5. Pure premium comparison on 2023 hold-out: independent vs joint

The copula correction reduces mean premiums by ~6% — reflecting that high-frequency
riders suppress claim size through NCD awareness.

---

## GAM Models with EBM (`02_gam_model.py`)

`InsuranceEBM` fits smooth non-linear shape functions per rating factor:

- Rider age: U-shaped relativity (range 2.5×)
- NCD years: monotone decreasing (range 2.0×)
- Region: London highest, Scotland lowest (range 1.7×)
- Engine cc: monotone increasing severity (range ~2.0×)

`RelativitiesTable` prints each factor's bins with raw score and relativity — directly
auditable by a pricing actuary without post-hoc SHAP.

---

## Ensemble + Conformal Intervals (`03_ensemble_pipeline.py`)

Equal-weight blend of GLM and EBM pure premiums, wrapped with
`InsuranceConformalPredictor`:

- Calibrated on 2022 hold-out with Pearson-weighted non-conformity score
- 90% prediction intervals on 2023 test year
- Distribution-free coverage guarantee regardless of model misspecification

---

## Bühlmann-Straub Credibility (`04_credibility.py`)

Region-level experience credibility blending. With 2 training years and 5 regions,
credibility factors Z range from ~0.58 (Scotland, smaller exposure) to ~0.81
(London, largest exposure). The regional credibility premiums materially correct
the flat prior — London blends 80% experience, 20% portfolio mean.

---

## Quote-to-Price Journey (`05_quote_to_price.py`)

For 5,000 new motorcycle quotes:

1. Technical pure premium from ensemble GLMs
2. Commercial gross-up: +12% expenses, +8% profit target, −5% online discount
3. NCD commercial discount (0–20%)
4. Regional market adjustment (competitive position)
5. +12% Insurance Premium Tax (IPT)

Resulting gross premiums: mean ~£353, range ~£100–£1,500+.

---

## Price-to-Sell Journey (`06_price_to_sell.py`)

Simulates a 10% discount A/B test on 8,000 quotes. `CausalPricingModel` (Double ML)
estimates the causal effect of the discount:

- **ATE ≈ +4pp** conversion uplift (statistically significant, p < 0.001)
- **ROI ≈ −59%** — discount costs £50/policy but earns back only ~£20 in revenue
- **CATE by NCD**: discount most effective for NCD 0–1 (price-sensitive new riders),
  near-zero for NCD 3–4 (loyal customers who would buy anyway)

---

## Production Monitoring (`07_monitoring.py`)

Monitors the 2021-trained GLM against 2022 and 2023:

- **Gini drift**: not significant (ranking stable)
- **GMCB calibration**: rejected — global level shift, recalibration recommended
- **A/E ratio**: ~0.60–0.64 (model over-predicts frequency)
- **PSI feature drift**: all features stable (PSI < 0.01)

Decision: `RECALIBRATE` — apply balance correction, no model refit required.

---

## Fairness & Governance (`08_fairness_governance.py`)

**Fairness audit** under UK Equality Act 2010:
- Age band and region as protected characteristics
- Demographic parity ratio flagged red (age is a rating factor — directionally
  expected but requires actuarial justification under FCA rules)

**MRM Model Card** records:
- Materiality tier 2 (annual independent validation required under PRA SS3/18)
- Three structured assumptions with HIGH/MEDIUM risk rating
- Monitoring triggers: Gini p-value, A/E ratio, PSI thresholds
- £8M GWP impacted
