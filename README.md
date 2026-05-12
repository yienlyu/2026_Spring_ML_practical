# 2026_Spring_ML_practical

## Overview
Goal:  explore the relationship between sleep quality and cardiovascular disease (CVD) risk, then use ML to predict who falls into the high-risk category.

## Step 1
Reads in sleep_health_dataset.csv — 100,000 records, each representing one person's sleep, physiological, and mental health data.

## Step 2
### Construct the Cardio Risk Index (CRI)
Because the dataset has no direct "cardiovascular disease" column, seven clinically grounded indicators are weighted and combined into a single proxy score:

| Indicator | Direction | Weight |
| --------- | --------- | ------ |
| Resting heart rate | Higher = more dangerous | 20% |
| BMI | Higher = more dangerous | 20% |
| Stress score | Higher = more dangerous | 25% |
| Sleep duration | Shorter = more dangerous | 15% |
| Deep sleep percentage | Lower = more dangerous | 10% |
| Wake episodes per night | More = more dangerous | 10% |
| Shift work | Adds a flat +0.5 if yes | - |

## Step 3
### Exploratory Data Analysis (EDA)
Visualizes the relationship between sleep metrics and CRI across three figures:

- Boxplots — compare the distribution of sleep quality, sleep duration, REM %, deep sleep %, etc. across the three risk groups
- Scatter plot — CRI vs. sleep quality score with a trend line; bar chart of mean CRI by sleep quality level
- Violin plot + bar chart — sleep disorder severity and mental health condition vs. mean CRI

## Step 4
### Statistical Testing
Confirms whether the differences between groups are genuinely significant or just random noise:

- Kruskal-Wallis test — a non-parametric equivalent of one-way ANOVA; tests whether all three groups differ
- Mann-Whitney U test — post-hoc pairwise comparison between Low and High risk groups, with effect size r calculated

## Step 5
### Machine Learning Classification
Reframes the problem as binary classification: predict whether a person belongs to the "High CVD Risk" group. Four models are trained:

1. Logistic Regression — linear baseline model
2. Decision Tree — interpretable rule-based model
3. Random Forest — ensemble of 200 trees, typically the strongest performer
4. SVM — trained on a 10,000-sample subset because the full dataset would be too slow

Models are evaluated on Accuracy, Weighted F1, and AUC-ROC.

## Step 6
### Model Visualization and Interpretation
- Confusion matrix — shows where Random Forest's predictions are right and wrong
- ROC curves — compares AUC across all four models
- Feature importance chart — which variables matter most for prediction
- LR coefficient plot — which factors increase vs. decrease the probability of high CVD risk

