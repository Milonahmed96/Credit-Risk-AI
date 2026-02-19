# Model Card: Credit Risk Classifier

## 1. Model Overview
* **Model Name:** Credit Risk Classifier (Baseline + Advanced)
* **Task:** Binary classification (Default vs. Non-default)
* **Domain:** Banking / Financial Risk Management
* **Target Variable:** Default payment next month (Yes / No)

This model predicts the probability that a credit card client will default in the next billing cycle. It is designed to support risk-aware decision-making, not automated approval.

## 2. Intended Use
**Primary Use**
* Credit risk assessment
* Loan approval decision support
* Risk-based customer monitoring

**Intended Users**
* Risk analysts
* Data scientists in banking / FinTech
* Credit policy teams

**Out-of-Scope Use**
* Fully automated loan approval without human oversight
* Use in jurisdictions with strict regulatory constraints without compliance review

## 3. Data Description
* **Source:** UCI Credit Card Default Dataset (Taiwan)
* **Size:** ~30,000 samples
* **Features include:**
  * Demographics (age, gender, education, marital status)
  * Credit limit
  * Payment history
  * Bill amounts and payment amounts
* **Target distribution:**
  * Non-default ≈ 78%
  * Default ≈ 22% (imbalanced)

## 4. Modeling Approach
**Baseline Model**
* Logistic Regression (class-weighted)
* Chosen for interpretability and regulatory alignment

**Advanced Model**
* Tree-based ensemble (XGBoost)
* Used to capture non-linear relationships and compounding financial behaviors

**Preprocessing**
* One-hot encoding for categorical features
* Scaling for continuous variables
* Ordinal handling for repayment status variables
* Engineered behavioral features (Utilization Ratio, Pay-to-Bill Ratio, Average Payment Delay, Bill Trend)

## 5. Evaluation Metrics
Because this is a risk-sensitive banking problem, accuracy alone is not sufficient. 
Reported metrics:
* ROC-AUC
* Precision (default class)
* Recall (default class)
* PR-AUC

Additionally: Cost-based threshold optimization was used to minimize financial loss.

## 6. Decision Threshold Strategy
Instead of a fixed 0.5 threshold, the model uses a business-driven threshold.
**Assumptions:**
* False Negative (approve defaulter) → high financial cost
* False Positive (reject good customer) → moderate opportunity cost

The optimal threshold was selected by minimizing total expected portfolio cost.

## 7. Explainability & Transparency
**Techniques Used:**
* Logistic regression coefficients
* Feature importance analysis
* SHAP (SHapley Additive exPlanations) values for local and global interpretability

**Key Risk Drivers Identified:**
* Previous payment delays (historical delinquency)
* Credit utilization patterns (credit exhaustion)
* High outstanding balances relative to payment history

Explainability enables regulatory transparency, risk team validation, and overall trust in model outputs.

## 8. Fairness & Ethical Considerations
This model may reflect historical biases present in the training data.
**Considerations:**
* Demographic features (like education or marital status) can correlate with default due to systemic factors.
* Approval rates across demographic groups should be continuously monitored.
* Model decisions should not be deployed without strict governance controls.
* No fully automated decisioning is recommended.

## 9. Limitations
* The dataset represents one specific geographic and temporal context (Taiwan, 2005).
* No macroeconomic variables (e.g., inflation, unemployment) are included.
* Concept drift is not modeled explicitly.
* Reject inference (the unknown behavior of rejected applicants) is not addressed.
* Results should not be assumed to generalize to modern portfolios without retraining and recalibration.

## 10. Monitoring & Maintenance (Conceptual)
In a production environment, the following must be monitored:
* Input feature drift
* Default rate changes (target drift)
* Performance decay over time
* Threshold recalibration needs

Retraining should be triggered periodically or upon statistical drift detection.

## 11. Future Improvements & Research Extensions
* Probability calibration (Platt scaling / Isotonic regression)
* Fairness-aware learning
* Reject inference techniques
* Time-aware risk modeling
* Integration with decision optimization systems

## 12. Model Versioning
* **Version:** v1.0
* **Date:** 2026-02-19
* **Author:** Milon Ahmed | Founder of Ahmed Intelligence
