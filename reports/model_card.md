# Model Card: Credit Risk Classifier

## 1. Model Overview
* **Model Name:** Credit Risk Classifier Pipelines (Baseline + Advanced)
* **Task:** Binary classification (Default vs. Non-default)
* **Domain:** Banking / Financial Risk Management
* **Target Variable:** Default payment next month (Yes / No)

This model predicts the probability that a credit card client will default in the next billing cycle. It is designed to support risk-aware decision-making using cost-optimized thresholds.

## 2. Intended Use
**Primary Use**
* Credit risk assessment
* Loan approval decision support
* Risk-based customer portfolio monitoring

**Out-of-Scope Use**
* Fully automated loan approval without human oversight
* Use in jurisdictions with strict regulatory constraints without compliance review

## 3. Modeling Approach & Architecture
**Pipeline Infrastructure**
* Both models utilize a strict `scikit-learn` Pipeline architecture to prevent data leakage.
* Missing values are handled dynamically via `SimpleImputer` (median for continuous, most frequent for categorical).

**Baseline Model**
* `LogisticRegressionCV` (class-weighted, 5-fold cross-validation)
* Chosen for interpretability and baseline regulatory alignment

**Advanced Model**
* Tree-based ensemble (`XGBoost`)
* Used to capture non-linear relationships and compounding financial behaviors

## 4. Evaluation Metrics
Reported metrics focus on risk-separation rather than raw accuracy:
* **Baseline ROC-AUC:** 0.7490
* **Advanced ROC-AUC:** 0.7803
* **Advanced PR-AUC:** 0.5561

## 5. Decision Threshold Strategy
Instead of a fixed 0.5 threshold, the model uses a business-driven threshold optimized via a cost matrix:
* False Negative (approve defaulter) cost: $5,000
* False Positive (reject good customer) cost: $1,000
* **Optimal Threshold Identified:** 0.42

## 6. Explainability & Transparency
**Techniques Used:**
* Logistic regression coefficients (reconstructed from pipeline outputs).
* SHAP (SHapley Additive exPlanations) values calculated on pipeline-transformed data.

**Key Risk Drivers Identified:**
1. `avg_payment_delay`: Historical delinquency over the past 6 months.
2. `pay_0`: Most recent payment status.
3. `utilization_ratio`: Current credit exhaustion.

## 7. Limitations & Governance
* The dataset represents one specific geographic and temporal context (Taiwan, 2005).
* No macroeconomic variables (e.g., inflation, unemployment) are included.
* Model decisions should not be deployed without strict human-in-the-loop governance controls.

## 8. Model Versioning
* **Version:** v2.0 (Pipeline Architecture Upgrade)
* **Date:** 2026-02-21
* **Author:** Milon Ahmed | Founder of Ahmed Intelligence
