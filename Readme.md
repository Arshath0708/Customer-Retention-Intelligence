# Customer Retention Intelligence System

Predicting and explaining customer churn for a retail bank — built to support
proactive retention decisions, not just generate a model score.

**[Live Demo →](https://customer-retention-intelligence-0708.streamlit.app/)**

## Business Problem

This bank loses ~20% of its customers annually. Each lost customer represents
an estimated €5,000 in lifetime value, while a retention outreach costs ~€150.
Currently there is no system to flag at-risk customers before they leave.

## What This Solves

A risk-scoring system that:
- Ranks all customers by churn probability
- Explains *why* each customer is at risk (not just a black-box score)
- Is tuned around the actual cost of false positives/negatives, not generic accuracy

## Key Findings (from EDA)

- Customers in **Germany churn at ~32%** vs ~16% in France/Spain — despite similar profiles
- **Inactive members churn at nearly 2x** the rate of active ones
- Counter-intuitively, customers holding **3-4 products churn more** than those with 1-2,
  suggesting over-selling rather than loyalty from cross-sell

## Approach

1. EDA to find behavioral drivers of churn (not just feature correlation)
2. Random Forest classifier, 5-fold cross-validated (ROC-AUC: 0.86)
3. **Threshold tuned to maximize estimated retention value**, not accuracy —
   moving from the default 0.5 cutoff to the business-optimal threshold (0.32) improved
   estimated annual value by ~€35,700 (raising total value saved to €396,900).
4. SHAP explainability so each flagged customer comes with a reason, not just a score
5. Deployed as a Streamlit tool with single-lookup and bulk CSV scoring

## Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.866 (CV: 0.862 ± 0.003) |
| Precision @ business threshold (0.32) | 37.4% |
| Recall @ business threshold (0.32) | 88.7% |
| Accuracy @ business threshold (0.32) | 67.5% |
| F1 Score @ business threshold (0.32) | 0.527 |
| Estimated annual retention value | €396,900 (Improvement of €35,700) |

## Tech Stack

Python · Pandas · Scikit-learn · SHAP · Streamlit

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What I'd Do With More Time

- A/B test the actual retention offers to validate the 30% assumed save rate
- Add a time-series component — churn risk likely changes month to month, not static
- Integrate directly with a CRM via API instead of CSV upload

## Dataset

[Bank Customer Churn Prediction — Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)