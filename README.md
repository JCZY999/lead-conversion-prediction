# Lead Conversion Prediction

An interactive Streamlit application that turns marketing engagement signals into an explainable lead-prioritization workflow. It is designed as a portfolio demonstration of the full path from marketing data to a usable ML decision tool.

**Live demo:** [Lead Conversion Prediction](https://lead-conversion-prediction-h2srqjegsu4ar2gyqvtduu.streamlit.app/)

## Program Enrollment Forecasting Case Study

The fully executed [Program-Level Student Enrollment Forecasting Case Study](program-enrollment-forecasting-case-study.ipynb) extends this project from lead-level prioritization to four-week-ahead enrollment planning by academic program. It connects Google keyword search demand, paid and organic traffic, GA4-style website behavior, CRM leads and applications, admissions outcomes, and enrollment.

The notebook includes a leakage-safe program-week ML workflow, staged funnel feature-set experiments, time-based validation, Ridge and tree-ensemble models, randomized tuning, MAE/RMSE/R²/WAPE/bias and interval coverage, program diagnostics, capacity-risk detection, planning-cost outcomes, interpretation, deployment design, monitoring, retraining, and incrementality testing.

## Business problem

Marketing and admissions/sales teams often receive more inquiries than they can follow up with immediately. A simple first-come, first-served workflow can delay response to high-intent leads and waste sales capacity on low-propensity records.

This app estimates each lead's conversion probability, assigns a follow-up priority, and translates the score into expected pipeline value. It is meant to support—not replace—sales judgment and lead-routing rules.

## Data used

The demo uses **synthetic lead-level data** to keep the project safe to share publicly. The schema reflects a realistic first-party marketing and CRM workflow:

| Data group | Example fields | Why it matters |
|---|---|---|
| Lead context | Lead ID, created date, region, acquisition source | Identifies where demand originates and enables channel-quality analysis. |
| Website behavior | Pages viewed, time on site, form started | Captures active research and high-intent site behavior. |
| Engagement | Email opens, prior lead probability, lead score | Incorporates ongoing nurture signals and historical context. |
| Outcome/value | Converted, expected pipeline USD | Supports supervised learning and business-value prioritization. |

Sources include Paid Search, Paid Social, SEO / Organic, Email / CRM, Display, and Referral. In production, these fields would be joined from GA4/GTM, paid-media platforms, Salesforce or HubSpot, and validated opportunity/application outcomes.

## Modeling approach

The portfolio workflow compares a transparent baseline with a higher-performing tree-based model:

1. **Logistic Regression baseline** for interpretable, calibrated comparison.
2. **XGBoost / LightGBM candidate model** for non-linear interactions between channel, engagement, and form behavior.
3. **Holdout evaluation** using ROC-AUC, PR-AUC, F1, precision/recall, and ranking lift.
4. **Operational scorecard** that converts probability into High Priority, Nurture, or Low Priority actions.

The live demo uses a transparent scoring function to make each interaction reproducible in a public app. A production implementation would train and persist the selected model from validated historical data, then monitor calibration, feature drift, and conversion-rate changes.

## Example evaluation

| Model | ROC-AUC | PR-AUC | F1 | Role |
|---|---:|---:|---:|---|
| Logistic Regression | 0.79 | 0.46 | 0.61 | Interpretable baseline |
| XGBoost / LightGBM | 0.86 | 0.58 | 0.67 | Selected candidate |
| Naïve baseline | 0.50 | 0.29 | 0.00 | Benchmark |

On the synthetic evaluation scenario, the highest-scored 20% of leads captures **61% of expected conversions**, representing approximately **2.7× lift versus random prioritization**. These are demo results, not claims about a live client or employer dataset.

## Business inference and measurable impact

- Route high-propensity leads to sales within one business hour while placing medium-intent leads in a personalized nurture sequence.
- Compare conversion propensity and expected pipeline across Paid Search, Paid Social, SEO, CRM, Display, and Referral instead of optimizing only for lead volume.
- Focus finite sales capacity on the top-ranked segment, measured by conversion capture, lift, speed-to-lead, pipeline created, and downstream opportunity/application conversion rate.
- Test whether score-based routing improves outcomes against the current process through a controlled holdout or phased rollout.

## Software and product stack

| Layer | Tools demonstrated |
|---|---|
| Data and ML | Python, Pandas, NumPy, scikit-learn, XGBoost / LightGBM concepts |
| Interactive product | Streamlit, Plotly, scenario controls, lead-level ranking, download-ready outputs |
| Deployment | GitHub repository + Streamlit Community Cloud |
| Production extension | SQL/BigQuery, FastAPI, Docker, GCP/Vertex AI, CI/CD, model monitoring, and data-quality tests |

## App pages

- **Score a Lead** — simulate an individual lead and see conversion probability, priority, expected pipeline, and local feature drivers.
- **Lead Intelligence** — compare channel quality, expected pipeline, and the highest-priority leads.
- **Model Evaluation** — review baseline versus candidate-model quality and ranking lift.
- **Methodology & Data** — understand the synthetic-data scope and production implementation path.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important note

All inputs, outcomes, model results, and business metrics in this repository are synthetic portfolio examples. The app should be retrained and validated against governed first-party data before use for operational decision-making.
