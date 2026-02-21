# 🏦 Credit Risk AI: Banking-Grade Loan Default Prediction

## Problem Statement
In retail banking, assessing credit risk is a high-stakes balancing act. The objective of this project is to build an interpretable, production-ready machine learning pipeline that predicts whether a credit card client will default in the next billing cycle. 

Crucially, this project moves beyond standard machine learning accuracy to focus on **banking-style decision-making**, optimizing for financial impact, risk mitigation, and regulatory explainability.

## Business Context
When evaluating a customer's creditworthiness, a bank faces two types of classification errors, each carrying vastly different financial weights:
* **False Negative (Approve a defaulter):** High financial loss (loss of principal balance and recovery costs, simulated at $5,000).
* **False Positive (Reject a good customer):** Opportunity loss and customer dissatisfaction (loss of future interest and transaction fees, simulated at $1,000).

👉 **The Banking Approach:** Because the cost of a default is significantly higher than the cost of a lost customer, a standard 0.5 decision threshold is financially dangerous. This project utilizes **cost-aware threshold tuning** to minimize total expected financial loss across the portfolio.

## Data & Feature Engineering
* **Dataset:** UCI Credit Card Default (Taiwan)
* **Target Variable:** `default` (Binary: 1 = Default, 0 = Non-Default)
* **Class Imbalance:** ~22% Default Rate.

Rather than relying on raw numerical data, banking-specific behavioral features were engineered to capture risk context:
* `utilization_ratio`: The bill amount normalized against the total credit limit (credit exhaustion).
* `pay_to_bill_ratio`: Identifying "transactors" (paying in full) versus "revolvers" (paying minimums).
* `avg_payment_delay`: A 6-month smoothed metric of historical delinquency.
* `bill_trend`: The velocity of debt accumulation over the 6-month period.

## Methodology & MLOps Architecture
To ensure zero data leakage and handle unexpected missing values in production, the models were built using strict `scikit-learn` Pipelines (`ColumnTransformer` -> `SimpleImputer` -> `StandardScaler`/`OneHotEncoder` -> `Estimator`).

* **Baseline Model:** Class-weighted `LogisticRegressionCV`. Tuned using 5-fold cross-validation for maximum baseline interpretability and regulatory alignment.
* **Advanced Model:** Pipeline-integrated XGBoost. Implemented to capture complex, non-linear financial behaviors and compounding risk factors.

## Evaluation & Results
Due to the 22% class imbalance, accuracy is a flawed metric. The models are evaluated strictly on their ability to rank-order risk:
* **Logistic Regression Baseline:** ROC-AUC: 0.7490
* **XGBoost Advanced:** ROC-AUC: 0.7803 | PR-AUC: 0.5561

**Cost Optimization:**
By simulating the business cost matrix, the algorithm identified **0.42** as the mathematically optimal probability threshold. Lowering the threshold from 0.50 to 0.42 allows the bank to preemptively flag higher-risk accounts, successfully minimizing total expected portfolio loss.

## Explainability
To comply with financial regulations and build trust with risk analysts, this model avoids the "black box" trap. 
* **Global Interpretability:** Logistic regression coefficients highlight directional risk (e.g., `avg_payment_delay` increases risk, while a higher `limit_bal` decreases risk).
* **Local Interpretability:** SHAP (SHapley Additive exPlanations) values were extracted dynamically from the XGBoost pipeline. The SHAP summary plot proves that engineered behavioral features—specifically historical payment delays and credit utilization ratios—are the primary drivers of model decisions, satisfying regulatory governance constraints.

***
*Author: Milon Ahmed | Founder of Ahmed Intelligence*
