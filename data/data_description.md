# Data Dictionary & Description

## Dataset Overview
* **Source:** UCI Machine Learning Repository (Default of Credit Card Clients Dataset - Taiwan)
* **Time Period:** April 2005 to September 2005
* **Total Observations:** 30,000
* **Target Variable:** `default` (Binary: 1 = Default next month, 0 = Non-default)

## 1. Demographic & Account Features
* **limit_bal:** Amount of given credit (in NT dollars). Includes both individual consumer credit and supplementary family credit.
* **sex:** Gender (1 = male; 2 = female).
* **education:** Educational background (1 = graduate school; 2 = university; 3 = high school; 4 = others). *Note: Undocumented values (0, 5, 6) were cleaned and mapped to 4.*
* **marriage:** Marital status (1 = married; 2 = single; 3 = others). *Note: Undocumented value (0) was cleaned and mapped to 3.*
* **age:** Age in years.

## 2. Historical Repayment Status (Ordinal)
The repayment status tracks delinquency over the past 6 months (April - September 2005). 
* **pay_0:** Repayment status in September 2005.
* **pay_2:** Repayment status in August 2005.
* **pay_3:** Repayment status in July 2005.
* **pay_4:** Repayment status in June 2005.
* **pay_5:** Repayment status in May 2005.
* **pay_6:** Repayment status in April 2005.

**Measurement Scale:** * -1 = Pay duly (paid on time)
* 1 = Payment delay for one month
* 2 = Payment delay for two months
* ...
* 8 = Payment delay for eight months
* 9 = Payment delay for nine months and above

## 3. Historical Bill Statements (Continuous)
The amount of the bill statement (in NT dollars) over the past 6 months.
* **bill_amt1:** Amount of bill statement in September 2005.
* **bill_amt2:** Amount of bill statement in August 2005.
* **bill_amt3:** Amount of bill statement in July 2005.
* **bill_amt4:** Amount of bill statement in June 2005.
* **bill_amt5:** Amount of bill statement in May 2005.
* **bill_amt6:** Amount of bill statement in April 2005.

## 4. Historical Payment Amounts (Continuous)
The amount paid by the client (in NT dollars) over the past 6 months.
* **pay_amt1:** Amount of previous payment in September 2005.
* **pay_amt2:** Amount of previous payment in August 2005.
* **pay_amt3:** Amount of previous payment in July 2005.
* **pay_amt4:** Amount of previous payment in June 2005.
* **pay_amt5:** Amount of previous payment in May 2005.
* **pay_amt6:** Amount of previous payment in April 2005.

---

## 5. Engineered Banking Features
To provide contextual business value, the following features were engineered from the raw data prior to modeling:

* **utilization_ratio:** `bill_amt1 / limit_bal` (Capped at lower=0). Measures how much of their total available credit the client is currently exhausting. High utilization is a strong indicator of financial distress.
* **pay_to_bill_ratio:** `pay_amt1 / bill_amt1` (Clipped between 0 and 1). Differentiates between "transactors" (clients who pay their balance in full) and "revolvers" (clients who only pay the minimum balance, carrying debt forward).
* **avg_payment_delay:** The mean of all `pay_X` columns (values < 0 clipped to 0). Provides a smoothed, 6-month historical view of the client's delinquency severity.
* **bill_trend:** `bill_amt1 - bill_amt6`. Captures the velocity of the client's debt accumulation or reduction over the 6-month observation window.
