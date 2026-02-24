# AI Credit Decision Engine: From Predictive ML to Production API

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## Problem Statement
In retail banking, assessing credit risk is a high-stakes balancing act. The objective of this project is to build an interpretable, production-ready machine learning pipeline that will both predicts whether a credit card client will default in the next billing cycle and build production decesion engine going beyond just applied data science into AI Engineering.

Crucially, this project moves beyond standard machine learning accuracy to focus on **banking-style decision-making**, optimizing for financial impact, risk mitigation, and regulatory explainability.

## Business Context
When evaluating a customer's creditworthiness, a bank faces two types of classification errors, each carrying vastly different financial weights:
* **False Negative (Approve a defaulter):** High financial loss (loss of principal balance and recovery costs, simulated at $5,000).
* **False Positive (Reject a good customer):** Opportunity loss and customer dissatisfaction (loss of future interest and transaction fees, simulated at $1,000).

**The Banking Approach:** Because the cost of a default is significantly higher than the cost of a lost customer, a standard 0.5 decision threshold is financially dangerous. This project utilizes **cost-aware threshold tuning** to minimize total expected financial loss across the portfolio.

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


## The AI Decision Engine
When a customer applies for a loan, they do not just hit the ML model. They pass through a simulated banking backend:

Hard Policy Check: Is utilization_ratio > 95%? Auto-Reject.
Hard Policy Check: Is avg_payment_delay > 2 months? Auto-Reject.
ML Risk Scoring: If policies pass, feed data to the XGBoost Pipeline.
Optimized Threshold Check: If ML Probability >= 0.42 -> Reject; Else -> Approve.

## How to Run the Live API Locally
Want to test the model like a Front-End Developer or Bank Manager? You can spin up the live API server on your machine in seconds.
Clone this repository and navigate to the folder:

git clone [https://github.com/Milonahmed96/Credit-Risk-AI.git](https://github.com/Milonahmed96/Credit-Risk-AI.git)
cd credit-risk-ai
python -m uvicorn src.app:app --reload

## Limitations & Ethical Considerations

Building a FinTech AI requires extreme scrutiny. While this system is robust, it operates under several assumptions and limitations that must be addressed before real-world deployment:

* **The Static Cost Matrix Assumption:** The threshold optimization assumes a fixed $5,000 loss for defaults and a $1,000 loss for false positives. In reality, credit limits and balances are highly dynamic. A production system should calculate cost dynamically *per applicant* based on their requested loan amount.
* **Concept Drift & Temporal Limitations:** The underlying dataset is based on Taiwan credit data from 2005. Economic conditions (inflation, interest rates, housing markets) shift continuously. A model trained on 2005 data will naturally degrade (concept drift) when predicting 2026 economic behaviors.

## Bias & Fairness (Fair Lending Compliance)
To maximize raw predictive power during the research phase, demographic features like `sex` and `marriage` were included in the dataset. However, in a real US or EU banking environment, utilizing these features violates the **Equal Credit Opportunity Act (ECOA)** and Fair Lending laws. 
* **The Fix:** Before deploying this to a real banking production environment, all protected demographic classes must be explicitly dropped from the training data, and the model must be audited using Disparate Impact analysis to ensure it does not unintentionally proxy these variables (e.g., zip codes proxying race).

## What Research Could Improve This
* **Macroeconomic Indicators:** The model currently only looks at micro-level customer behavior. Incorporating macro-features like the current national unemployment rate, inflation index, and federal interest rates would make the model resilient to market crashes.
* **Survival Analysis:** Moving beyond binary classification (Will they default? Yes/No) to Survival Analysis (At what specific month will they default?) would allow the bank to intervene proactively before the default occurs.

## How the System Could Evolve (V2 Architecture)
To scale this from a local API to an enterprise-grade banking application, the following infrastructure upgrades are required:
1. **Containerization:** Wrapping the FastAPI server in **Docker** and deploying it via Kubernetes (EKS/GKE) to handle thousands of concurrent loan applications.
2. **MLflow Model Registry:** Implementing MLflow to track model versions and trigger automated retraining pipelines when data drift is detected.
3. **Cloud Database:** Transitioning the JSON compliance Audit Log to a secure, encrypted PostgreSQL database on AWS RDS for permanent regulatory storage.

## Tech Stack Used

Core Machine Learning: scikit-learn (Pipelines, Preprocessing), XGBoost.
Explainable AI (XAI): SHAP (Global and Local Interpretability).
Engineering & Deployment: FastAPI, Uvicorn, Pydantic, joblib.
Data & Math: pandas, numpy.

*Author: Milon Ahmed | Founder of Ahmed Intelligence*
