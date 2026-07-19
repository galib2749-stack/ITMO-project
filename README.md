# Uplift Modeling with Sequential Customer Behavior Representations

### Оценка приростного эффекта клиентских коммуникаций с использованием последовательности пользовательского поведения

Author: **Галиб Байрамов** — submission for the ИТМО AI Talent Hub Junior ML
Contest ([ai.itmo.ru/junior_ml_contest](https://ai.itmo.ru/junior_ml_contest)).

Dataset: **X5 RetailHero Uplift Modeling** (real, public;
[ods.ai/competitions/x5-retailhero-uplift-modeling](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data)).

## What this is

A from-scratch uplift-modeling project comparing four required approaches
for deciding which loyalty clients should receive a marketing SMS:

0. Random targeting (baseline)
1. Response CatBoost (`P(Y=1|X)` — shown to NOT be an uplift model)
2. T-Learner (two CatBoost classifiers)
3. X-Learner (full 4-stage Kunzel et al. 2019 formulation)
4. **Transformer Encoder with Shared Customer Representation and Two
   Outcome Heads** — trained on raw purchase-receipt sequences, not only
   aggregated features

All five are evaluated on the same held-out split with the same metric
implementations (Qini coefficient, AUUC, uplift@10/20/30%, bootstrap CIs).

## Where to start

- **`notebooks/x5_uplift_full_pipeline.ipynb`** — the single master notebook.
  Runs top to bottom from a clean kernel; contains every stage (audit,
  leakage analysis, split, feature/sequence engineering, all 5
  models, evaluation, business scenario, A/B design, limitations,
  conclusions).
- `reports/data_download_report.md` — dataset provenance and checksums.
- `reports/leakage_audit.md` — a concrete leakage finding (raw
  `first_redeem_date` is contaminated by post-campaign events; the official
  example baseline uses it uncensored, this project does not).
- `docs/features.md` — full feature documentation and exclusion rationale.
- `docs/itmo_requirements.md` — contest requirements mapping.
- `artifacts/metrics.csv` — final metrics table for all 5 required rows.
- `reports/experiment_report.md` — written results summary.
- `application/` — CV, motivation letter, project description (PDF + md).
- `presentation/` — slide deck, speaker notes, script, Q&A.

## Project layout

```
data/raw/x5_retailhero/     real dataset (clients, products, purchases, uplift_train/test)
data/interim/                streaming-pipeline outputs (per-client aggregates, receipt sequences)
data/processed/               train/val/holdout split, cached prediction arrays
src/                          all pipeline code (imported by the notebook, not duplicated)
tests/                        unit tests for uplift metrics
artifacts/                    metrics, trained Transformer weights/config, qini curve data
reports/figures/              all generated plots
notebooks/                    the master notebook
docs/                         requirements mapping, feature documentation
application/                  CV, motivation letter, project description
presentation/                 slide deck and supporting speaker materials
scripts/                      build tooling: notebook builder, presentation builder,
                               markdown->PDF renderer, PowerPoint COM slide-render/QA script
"ИТМО AI Talent Hub"/         final submission bundle for the contest platform
```

## Reproducing

```bash
pip install -r requirements.txt   # numpy, pandas, catboost, torch, scikit-learn, pyarrow, matplotlib, python-pptx, reportlab, pytest
python -m pytest tests/ -v
python src/run_full_pipeline.py 15      # regenerates artifacts/metrics.csv etc. from scratch
# Note: run nbconvert from the project root and pin ExecutePreprocessor.cwd to it --
# the notebook uses paths relative to the project root (e.g. data/raw/...), not to
# notebooks/, and nbconvert's default execution cwd is the notebook's own folder.
python -m nbconvert --to notebook --execute notebooks/x5_uplift_full_pipeline.ipynb \
  --output x5_uplift_full_pipeline_executed.ipynb --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.cwd="$(pwd)"
python -m nbconvert --to html notebooks/x5_uplift_full_pipeline_executed.ipynb \
  --output ../reports/x5_uplift_full_pipeline.html
```

## Honesty notes

- All modeling results are computed on the **real** X5 RetailHero dataset —
  synthetic data was used only for smoke-testing code during development,
  never for reported metrics (see `reports/data_download_report.md`).
- Business-impact figures (`Section 16` of the notebook) are an explicit,
  labeled **scenario** (illustrative margin/cost assumptions), not measured
  X5 or T-Bank financial data.
- This project was built with **Claude Code** (Anthropic) as an AI
  pair-programming agent under the author's direction — disclosed, not
  hidden, per the contest's "AI Application" evaluation criterion.
