# Policy Optimization for Financial Decision-Making
*Name:* Ankit Aggarwal
*Date:* 30/10/2025

## 1. Project Overview & Objective

The goal of this project was to compare two machine learning paradigms for a fintech loan approval process:
1.  *Supervised DL:* A predictive model to estimate the probability of default.
2.  *Offline RL:* A decision-making agent to learn an optimal approval policy to maximize financial return.

I used a sample of 10000 loans from the LendingClub dataset, focusing on loans with a final status of "Fully Paid" or "Charged Off."

---

## 2. Model 1: Predictive Deep Learning Model

### 2.1. Model & Metrics
* *Architecture:* I built a Multi-Layer Perceptron (MLP) using TensorFlow/Keras with [Number] hidden layers.
* *Target:* Predict is_default (1 for "Charged Off", 0 for "Fully Paid").
* *Key Challenge:* The dataset was imbalanced, with [X]% non-defaulters. I addressed this using class_weight='balanced' during training.

### 2.2. Final Results (Test Set)
* *Area Under the ROC Curve (AUC):* 0.7086
* *F1-Score:* 0.4218

### 2.3. Analysis of Metrics
* *Why AUC/F1?* These metrics are ideal for a predictive classifier task.
* *AUC* measures the model's ability to distinguish between a "good" loan and a "bad" loan across all possible thresholds. An AUC of [Your AUC] indicates [a good/fair/poor] ability to separate classes.
* *F1-Score* measures the balance between Precision and Recall, which is crucial for an imbalanced dataset. It tells us how effective the model is at identifying the minority class (defaulters) without incorrectly flagging too many good loans.

These metrics tell us *"How well does the model *identify risk?"** They do not tell us what to do with that risk information to make money.

---

## 3. Model 2: Offline Reinforcement Learning Agent

### 3.1. RL Problem Formulation
* *State (s):* The preprocessed vector of applicant features (e.g., loan_amnt, grade, dti).
* *Action (a):* A discrete space of {0: Deny Loan, 1: Approve Loan}.
* *Reward (r):* This was engineered to reflect the business goal:
    * If Approve & Fully Paid: reward = +(loan_amnt * int_rate)
    * If Approve & Defaulted: reward = -loan_amnt
* *Algorithm:* I used *Conservative Q-Learning (CQL)* from the d3rlpy library, as it is designed for offline datasets where we only have data for a single action (in our case, action=1).

### 3.2. Final Results (Test Set)
* *Estimated Policy Value (from FQE):* 1924.2563

### 3.3. Analysis of Metric
* *Why Estimated Policy Value?* This metric directly answers the business question.
* It represents the *expected financial return (average profit/loss) per loan* if we were to deploy this RL agent's policy in the real world.
* A positive value of 1924.2563 suggests the policy is, on average, profitable, while a negative value would suggest it loses money. This metric combines the probability of an outcome (which the DL model predicts) with the magnitude of that outcome (the financial reward/loss), which is what the business truly cares about.

---

## 4. Comparative Analysis & Future Steps

### 4.1. Policy Disagreements
The most insightful part of this project was analyzing where the two models disagreed.

I found 7777 cases where the *DL model would "Deny" but the RL agent would "Approve."*

*Example Analysis:*
* I observed that these applicants often had a *high interest rate* (e.g., > 15%) and a *high predicted probability of default* (e.g., > 0.5).
* *DL Model's Logic:* The model saw the high default probability and correctly flagged it as "high-risk," leading to a "Deny" action.
* *RL Agent's Logic:* The agent saw the same high risk but also saw the *high potential reward* from the high interest rate. It learned that even with a 60% chance of default, the 40% chance of a large interest-based profit made the "Approve" action have a higher expected value than the "Deny" action (which has a guaranteed reward of 0).

The RL agent learned a more nuanced, profit-maximizing policy that is willing to take calculated risks for high rewards, whereas the simple DL classifier only identified the risk.

### 4.2. Limitations
1.  *Selection Bias:* The entire dataset consists of accepted loans. We have zero data on what happens when a loan is denied (the counterfactual). The RL agent assumes a "Deny" action gives a reward of 0, but this might not be true (e.g., opportunity cost, or that applicant applies again later).
2.  *Simple Reward:* My reward function was simple. It didn't account for the time value of money or the cost of capital.
3.  *Static Dataset:* The world changes. A policy trained on 2007-2018 data may not perform well in a 2025 economic climate.

### 4.3. Future Steps
* *Deploy as an A/B Test:* I would *not* deploy this RL agent to replace the entire system. I would deploy it as an A/B test on a small fraction (e.g., 5%) of new applications. We could run the RL policy and the existing policy in parallel to see which one performs better on new, live data.
* *Explore Counterfactuals:* I would want to gather data on rejected applicants to build a more robust model of the true action space.

* *Refine the Threshold:* For the DL model, the 0.5 threshold is arbitrary. I would run simulations to find the optimal probability threshold that maximizes business value, effectively turning the DL model into a "policy" that can be compared more directly to the RL agent.
