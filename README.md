# explore_burning_cost_mc

Full-lifecycle UK motorcycle insurance pricing: synthetic data, an
ensemble of GLM and gradient-boosted pricing models, actuarial rating
relativities, and a market simulation carrying a policy from quote through
bind through renewal.

Built on top of [`burning_cost`](https://github.com/burningcost/burning-cost)
for exposure calculation, claims aggregation, and SHAP-based rating
relativities — the parts of that project that are an actual, working
Python library rather than its AI-agent orchestration tooling (see
[*A note on the `burning-cost` name*](#a-note-on-the-burning-cost-name)
below before you go looking for it on PyPI).

## What's here

```
src/moto_pricing/
  data/
    synthetic.py      # UK motorcycle policy + claims generator, known DGP
    build_dataset.py  # wires the raw tables through burning_cost's
                       # calculate_exposure() and aggregate_claims()
  features.py          # rating factor lists, out-of-time train/test split
  models/
    glm.py             # Poisson frequency / Gamma severity GLM baseline
    gbm.py              # CatBoost frequency/severity ensemble
    ensemble.py          # GLM x GBM blend, weights chosen on a held-out year
  relativities.py       # burning_cost.shap_relativities -> rate tables
  evaluation.py          # Gini, actual-vs-expected by decile
  pricing.py              # risk premium -> gross premium -> IPT
  journey/
    quotes.py             # price prospects, simulate a competing market price
    conversion.py           # price-elasticity new-business conversion model
    lifecycle.py             # NCD step-up/down, re-rate, renewal retention
  pipeline.py                # ties all of the above together
scripts/run_pipeline.py       # CLI entry point
tests/                         # pytest, ~28 tests covering every stage
```

## The pipeline, stage by stage

1. **Synthetic book** (`data/synthetic.py`): a policy table and a separate
   claims table generated from a known Poisson-frequency / Gamma-severity
   data generating process, calibrated to real UK motorcycle market
   patterns — young riders on high-cc sports bikes carry sharply higher
   frequency *and* severity; theft risk depends on security devices and
   overnight parking; NCD is strongly protective.
2. **Modelling dataset** (`data/build_dataset.py`): `burning_cost.data.exposure.calculate_exposure`
   turns policy dates into earned duration (handling leap years and
   mid-term cancellations correctly), and `burning_cost.data.aggregation.aggregate_claims`
   joins the claims table onto it.
3. **Pricing models** (`models/`): a Poisson/Gamma GLM baseline, a small
   CatBoost ensemble (two hyperparameter variants per target, correctly
   using `baseline=log(exposure)` as an offset rather than a sample
   weight — the single most common mistake in GBM frequency modelling),
   and a blend of the two with weights chosen on a held-out training year.
   All three are evaluated out-of-time on 2023 business.
4. **Rating relativities** (`relativities.py`): `burning_cost.shap_relativities.SHAPRelativities`
   converts the CatBoost frequency model's SHAP values into the same
   (feature, level, multiplicative relativity) shape a GLM's `exp(beta)`
   table would give you — interpretable enough for a pricing committee,
   built from a model with no linearity assumption.
5. **Pricing loadings** (`pricing.py`): risk premium to gross premium via
   fixed expenses, variable expenses, commission and profit margin, then
   UK Insurance Premium Tax on top.
6. **Quote → bind** (`journey/quotes.py`, `journey/conversion.py`): a fresh
   batch of prospects is priced by the fitted ensemble; a simulated
   competitor market price is generated from the *true* risk (the DGP,
   not the fitted model); conversion is a logistic function of how
   competitive our quote is.
7. **Renewal** (`journey/lifecycle.py`): bound policies generate a year of
   claims experience, NCD steps up or down accordingly, the policy is
   re-rated a year older, and a retention model (strong inertia, price
   and claims-experience sensitive) decides who renews. Renewal pricing
   deliberately uses the *same* model and loadings as new business — since
   the FCA's January 2022 general insurance pricing practices rules, an
   insurer isn't allowed to charge a loyal renewing customer more than an
   equivalent new customer, so this pipeline doesn't either.

## Running it

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run scripts/run_pipeline.py --n-policies 150000 --n-prospects 40000
```

This writes `outputs/pipeline_report.md`, a model-comparison table, the
SHAP relativity tables, and a handful of charts (Gini comparison,
actual-vs-expected, relativity bar charts, the quote-to-bind-to-renewal
funnel). Both sizes are configurable — the numbers above generate roughly
150k historical policies and 40k fresh quotes, which takes a few minutes.

Run the test suite with:

```bash
uv run pytest
```

## A note on the `burning-cost` name

Worth flagging for anyone reusing this: the GitHub repository
[`burningcost/burning-cost`](https://github.com/burningcost/burning-cost)
(no hyphen in the org name) is what this project depends on — it has the
real `calculate_exposure`, `aggregate_claims` and `SHAPRelativities` code
used above, pulled in directly via a `git+https://...` dependency in
`pyproject.toml`.

Separately, **PyPI has a package literally named `burning-cost`** (hyphen,
different org) that is a "meta-package" whose README instructs installing
ten further packages (`insurance-causal`, `insurance-conformal`,
`insurance-credibility`, etc.) that this project does not use, has not
reviewed, and does not vouch for. If you're setting this up yourself,
don't `pip install burning-cost` expecting to get the code described
above — you won't. Get it from the GitHub repository, as this project's
`pyproject.toml` does.
