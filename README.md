# Credit Risk AI: Banking-Grade Loan Default Prediction

## Problem Statement
In retail banking, assessing credit risk is a high-stakes balancing act. The objective of this project is to build an interpretable machine learning model that predicts whether a credit card client will default in the next billing cycle. 

Crucially, this project moves beyond standard machine learning accuracy to focus on **banking-style decision-making**, optimizing for financial impact, risk mitigation, and regulatory explainability.

## Business Context
When evaluating a customer's creditworthiness, a bank faces two types of classification errors, each carrying vastly different financial weights:
* **False Negative (Approve a defaulter):** High financial loss (loss of principal balance and recovery costs).
* **False Positive (Reject a good customer):** Opportunity loss and customer dissatisfaction (loss of future interest and transaction fees).

**The Banking Approach:** Because the cost of a default is significantly higher than the cost of a lost customer, a standard 0.5 decision threshold is financially dangerous. This project utilizes **cost-aware threshold tuning** to minimize total expected financial loss across the portfolio.

## Data Description
* **Dataset:** UCI Credit Card Default (Taiwan)
* **Target Variable:** `default.payment.next.month` (Binary: 1 = Default, 0 = Non-Default)
* **Samples:** ~30,000 historical credit card clients.
* **Class Imbalance:** ~22% Default Rate.
* **Features:** A combination of demographic indicators (Age, Education, Marital Status) and highly predictive financial behaviors (6-month repayment history, billing amounts, and payment amounts).

## Methodology

### 1. Advanced Feature Engineering
Rather than relying on raw numerical data, banking-specific behavioral features were engineered to capture risk context:
* `utilization_ratio`: The bill amount normalized against the total credit limit (credit exhaustion).
* `pay_to_bill_ratio`: Identifying "transactors" (paying in full) versus "revolvers" (paying minimums).
* `avg_payment_delay`: A 6-month smoothed metric of historical delinquency.
* `bill_trend`: The velocity of debt accumulation over the 6-month period.

### 2. Modeling Progression
* **Baseline Model:** Class-weighted Logistic Regression. Chosen for its baseline interpretability, stable coefficients, and strict regulatory alignment.
* **Advanced Model:** XGBoost (Gradient Boosting). Implemented to capture complex, non-linear financial behaviors and compounding risk factors.

## Evaluation & Results
Due to the 22% class imbalance, accuracy is a flawed and misleading metric. The models are evaluated strictly on their ability to manage risk using:
* **ROC-AUC & PR-AUC:** Measuring the model's ability to rank-order risk across all thresholds.
* **Precision/Recall Trade-off:** Focusing heavily on the recall of the default class.
* **Threshold vs. Cost Optimization:** Simulating a business cost matrix to find the exact probability threshold that minimizes total portfolio loss.



**Key Takeaway:** At our optimized decision threshold, we drastically reduce financial loss compared to the default 0.5 threshold, proving the tangible value of cost-based machine learning in banking.

## Explainability
To comply with financial regulations and build trust with risk analysts, this model avoids the "black box" trap. 
* **Global Interpretability:** Logistic regression coefficients are extracted to highlight directional risk (e.g., late payments drive risk up, while high credit limits drive risk down).
* **Local Interpretability:** SHAP (SHapley Additive exPlanations) values are utilized on the tree-based models to explain exactly *why* a specific customer was flagged for default, ranking features by their direct impact on the model's output.



## Limitations
* **Macroeconomic Factors:** The dataset represents a specific geographic and temporal context (Taiwan) and lacks macroeconomic indicators like unemployment rates, inflation, or interest rate changes.
* **Static Snapshot:** Concept drift is not explicitly modeled; real-world deployment would require continuous monitoring of shifting financial behaviors.

## Future Work (Research Extensions)
* **Reject Inference:** Adjusting the model to account for the unobserved performance of applicants who were rejected.
* **Probability Calibration:** Using Platt Scaling or Isotonic Regression to ensure output probabilities perfectly match real-world, observed default rates.
* **Fairness-Aware Learning:** Implementing algorithmic constraints to ensure demographic parity and eliminate systemic bias in automated decisions.
* **Concept Drift Monitoring:** Building a pipeline to detect when consumer credit behavior fundamentally shifts away from the training data distribution.

***
*Author: Milon Ahmed | Founder of Ahmed Intelligence*
