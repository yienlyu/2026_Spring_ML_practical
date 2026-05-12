# 2026_Spring_ML_practical

## Overview

Goal: Use a 100,000-record sleep health dataset to build two legitimate machine learning tasks — both targeting real columns in the dataset with no circular logic. That is, predict sleep disorder risk (Task A), and predict cognitive performance score (Task B).

Goal EXTRA: explore the relationship between sleep quality and cardiovascular disease (CVD) risk, then use ML to predict who falls into the high-risk category.

## Task A: Predict Sleep Disorder Risk (Classification)

#### Target: `sleep_disorder_risk` (Healthy / Mild / Moderate / Severe)

Four models are trained and compared:

| Model               | Key Design Choice                                                         |
| ------------------- | ------------------------------------------------------------------------- |
| Logistic Regression | Linear baseline; uses StandardScaler inside a Pipeline                    |
| Decision Tree       | `max_depth=8`; `class_weight='balanced'` to handle the 4% Severe minority |
| Random Forest       | 300 trees; ensemble voting reduces overfitting                            |
| SVM (RBF kernel)    | Trained on a 12,000-sample subset because full SVM is O(n²)               |

Evaluated on Accuracy, Weighted F1, and AUC-OVR (One-vs-Rest).

## Task B: Predict Cognitive Performance Score (Regression)

#### Target: `cognitive_performance_score` (continuous, 0–100)

| Model                   | Key Design Choice                                      |
| ----------------------- | ------------------------------------------------------ |
| Linear Regression       | Straight-line baseline                                 |
| Decision Tree Regressor | Leaf nodes output mean values instead of class labels  |
| Random Forest Regressor | 300 trees averaged together                            |
| SVR (RBF kernel)        | `epsilon=0.5` tolerance band; subset of 12,000 samples |

Evaluated on RMSE, MAE, and $R^2$.

## Feature Design (No Data Leakage)

Features are strictly drawn from behavioral, physiological, sleep-objective, and environmental columns. Target columns are never used as inputs, so accuracy numbers are genuinely meaningful — unlike the original CRI-based analysis.

## Model Interpretation (13 charts total)

Each task produces a full set of interpretation charts:

- Confusion matrix (raw + normalized) — where the classifier goes right and wrong
- ROC curves — per-class AUC for the classifier
- Residual plots — whether regression errors are random or systematic
- Predicted vs. Actual — how closely predictions track ground truth
- Gini feature importance — which features the Random Forest used most during training
- Permutation importance — which features actually hurt performance when shuffled (more reliable than Gini)
- Decision tree visualization (depth=3) — a human-readable view of the model's decision logic
- Logistic Regression coefficients — which features push each class probability up or down

## Bottom Line

Random Forest is the strongest model in both tasks. The most influential features across both targets are sleep quality score, sleep duration, stress score, and deep sleep percentage — consistent with the clinical literature on sleep and health outcomes.

## EXTRA: Predict CVD Risk

### Step 1

Reads in sleep_health_dataset.csv, each representing one person's sleep, physiological, and mental health data.

### Step 2

#### Construct the Cardio Risk Index (CRI)

Because the dataset has no direct "cardiovascular disease" column, seven clinically grounded indicators are weighted and combined into a single proxy score:

| Indicator               | Direction                | Weight |
| ----------------------- | ------------------------ | ------ |
| Resting heart rate      | Higher = more dangerous  | 20%    |
| BMI                     | Higher = more dangerous  | 20%    |
| Stress score            | Higher = more dangerous  | 25%    |
| Sleep duration          | Shorter = more dangerous | 15%    |
| Deep sleep percentage   | Lower = more dangerous   | 10%    |
| Wake episodes per night | More = more dangerous    | 10%    |
| Shift work              | Adds a flat +0.5 if yes  | -      |

### Step 3

#### Exploratory Data Analysis (EDA)

Visualizes the relationship between sleep metrics and CRI across three figures:

- Boxplots — compare the distribution of sleep quality, sleep duration, REM %, deep sleep %, etc. across the three risk groups
- Scatter plot — CRI vs. sleep quality score with a trend line; bar chart of mean CRI by sleep quality level
- Violin plot + bar chart — sleep disorder severity and mental health condition vs. mean CRI

### Step 4

#### Statistical Testing

Confirms whether the differences between groups are genuinely significant or just random noise:

- Kruskal-Wallis test — a non-parametric equivalent of one-way ANOVA; tests whether all three groups differ
- Mann-Whitney U test — post-hoc pairwise comparison between Low and High risk groups, with effect size r calculated

### Step 5

#### Machine Learning Classification

Reframes the problem as binary classification: predict whether a person belongs to the "High CVD Risk" group. Four models are trained:

1. Logistic Regression — linear baseline model
2. Decision Tree — interpretable rule-based model
3. Random Forest — ensemble of 200 trees, typically the strongest performer
4. SVM — trained on a 10,000-sample subset because the full dataset would be too slow

Models are evaluated on Accuracy, Weighted F1, and AUC-ROC.

### Step 6

#### Model Visualization and Interpretation

- Confusion matrix — shows where Random Forest's predictions are right and wrong
- ROC curves — compares AUC across all four models
- Feature importance chart — which variables matter most for prediction
- LR coefficient plot — which factors increase vs. decrease the probability of high CVD risk
