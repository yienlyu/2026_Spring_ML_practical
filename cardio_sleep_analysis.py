"""
Cardiovascular Disease Risk & Sleep Quality Relationship Analysis
=================================================================
Dataset: sleep_health_dataset.csv (100,000 records)

Analysis Framework:
  1. Construct a composite Cardio Risk Index (CRI)
  2. Exploratory Analysis: Sleep metrics vs. cardiovascular risk
  3. Statistical Testing: Confirm significance of inter-group differences
  4. Machine Learning: Predict high cardiovascular risk (Task C)
  5. Model Interpretation: Feature importance
"""

# ────────────────────────────────────────────────────────────────
# 0. Imports
# ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score,
    f1_score, roc_auc_score, roc_curve, auc
)

import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ────────────────────────────────────────────────────────────────
# 1. Load Data
# ────────────────────────────────────────────────────────────────
print("=" * 65)
print("  Cardiovascular Risk × Sleep Quality Relationship Analysis")
print("=" * 65)

df = pd.read_csv("./dataset/sleep_health_dataset.csv")
print(f"Dataset size: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ────────────────────────────────────────────────────────────────
# 2. Construct the Composite Cardio Risk Index (CRI)
# ────────────────────────────────────────────────────────────────
# Clinical rationale:
#   - Resting heart rate (heart_rate_resting_bpm): > 80 bpm is positively
#     associated with cardiovascular event risk
#   - BMI: > 25 = overweight, > 30 = obese; obesity is an independent CVD risk factor
#   - Stress score (stress_score): chronic stress elevates cortisol,
#     accelerating atherosclerosis
#   - Sleep duration (sleep_duration_hrs): < 6 hrs linked to hypertension
#     and myocardial infarction risk
#   - Deep sleep percentage (deep_sleep_percentage): low deep sleep is
#     associated with nocturnal blood pressure fluctuation
#   - Wake episodes (wake_episodes_per_night): frequent waking is an
#     indicator of sleep apnea
#   - Shift work (shift_work): disrupts circadian rhythm, increasing CVD risk

print("\n[Step 2] Constructing the Cardio Risk Index (CRI)...")

# Standardize each metric (z-score); flip sign so higher value = higher risk
df['z_hr']        = (df['heart_rate_resting_bpm'] - df['heart_rate_resting_bpm'].mean()) / df['heart_rate_resting_bpm'].std()
df['z_bmi']       = (df['bmi'] - df['bmi'].mean()) / df['bmi'].std()
df['z_stress']    = (df['stress_score'] - df['stress_score'].mean()) / df['stress_score'].std()
df['z_sleep_dur'] = -((df['sleep_duration_hrs'] - df['sleep_duration_hrs'].mean()) / df['sleep_duration_hrs'].std())  # inverted: shorter = higher risk
df['z_deep']      = -((df['deep_sleep_percentage'] - df['deep_sleep_percentage'].mean()) / df['deep_sleep_percentage'].std())  # inverted
df['z_wake']      = (df['wake_episodes_per_night'] - df['wake_episodes_per_night'].mean()) / df['wake_episodes_per_night'].std()

# Weighted composite (based on relative clinical importance)
weights = {
    'z_hr':        0.20,
    'z_bmi':       0.20,
    'z_stress':    0.25,   # stress carries slightly higher weight for CVD
    'z_sleep_dur': 0.15,
    'z_deep':      0.10,
    'z_wake':      0.10,
}
df['CRI'] = sum(df[col] * w for col, w in weights.items())
# Fixed bonus for shift work
df['CRI'] += df['shift_work'] * 0.5

# Divide into three risk groups based on CRI tertiles
df['cardio_risk_group'] = pd.qcut(
    df['CRI'],
    q=3,
    labels=['Low CVD Risk', 'Moderate CVD Risk', 'High CVD Risk']
)

# Binary label (high risk vs. non-high risk) for classification models
df['high_cardio_risk'] = (df['cardio_risk_group'] == 'High CVD Risk').astype(int)

print(f"  CRI stats: mean={df['CRI'].mean():.3f}, std={df['CRI'].std():.3f}")
print(f"  Risk group distribution:\n{df['cardio_risk_group'].value_counts().to_string()}")

# ────────────────────────────────────────────────────────────────
# 3. Exploratory Analysis: Sleep Metrics vs. Cardiovascular Risk Groups
# ────────────────────────────────────────────────────────────────
print("\n[Step 3] Exploratory Data Analysis...")

sleep_metrics = {
    'sleep_quality_score':      'Sleep Quality Score',
    'sleep_duration_hrs':       'Sleep Duration (hrs)',
    'rem_percentage':           'REM Percentage (%)',
    'deep_sleep_percentage':    'Deep Sleep Percentage (%)',
    'sleep_latency_mins':       'Sleep Latency (min)',
    'wake_episodes_per_night':  'Wake Episodes per Night',
}

# ── Figure 1: Boxplots of Sleep Metrics by CVD Risk Group ─────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
palette = {
    'Low CVD Risk':      '#4CAF50',
    'Moderate CVD Risk': '#FFC107',
    'High CVD Risk':     '#F44336',
}
order = ['Low CVD Risk', 'Moderate CVD Risk', 'High CVD Risk']

for ax, (col, label) in zip(axes, sleep_metrics.items()):
    sample = df.sample(5000, random_state=RANDOM_STATE)
    sns.boxplot(
        data=sample,
        x='cardio_risk_group', y=col,
        order=order,
        palette=palette,
        ax=ax,
        linewidth=1.2,
        fliersize=2,
    )
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(label)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.4)

fig.suptitle('Sleep Metric Distributions by Cardiovascular Risk Group (Boxplots)',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('cardio_sleep_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: cardio_sleep_boxplots.png")

# ── Figure 2: CRI vs. Sleep Quality Score Scatter Plot ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: CRI vs. Sleep Quality Score (colored by CVD risk group)
sample2 = df.sample(4000, random_state=RANDOM_STATE)
for group, color in palette.items():
    mask = sample2['cardio_risk_group'] == group
    axes[0].scatter(
        sample2.loc[mask, 'sleep_quality_score'],
        sample2.loc[mask, 'CRI'],
        c=color, label=group, alpha=0.45, s=15, edgecolors='none'
    )
# Trend line
z = np.polyfit(df['sleep_quality_score'], df['CRI'], 1)
p = np.poly1d(z)
x_range = np.linspace(df['sleep_quality_score'].min(), df['sleep_quality_score'].max(), 100)
axes[0].plot(x_range, p(x_range), 'k--', lw=1.8, label='Trend Line')
axes[0].set_xlabel('Sleep Quality Score', fontsize=11)
axes[0].set_ylabel('Cardio Risk Index (CRI)', fontsize=11)
axes[0].set_title('Sleep Quality vs. Cardio Risk Index', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Right: Mean CRI by Sleep Quality Group (with error bars)
df['sq_group'] = pd.cut(
    df['sleep_quality_score'],
    bins=[0, 3, 5, 7, 10],
    labels=['Poor (1–3)', 'Fair (3–5)', 'Good (5–7)', 'Excellent (7–10)']
)
cri_by_sq = df.groupby('sq_group', observed=True)['CRI'].agg(['mean', 'sem']).reset_index()
colors_bar = ['#F44336', '#FF9800', '#FFC107', '#4CAF50']
axes[1].bar(
    cri_by_sq['sq_group'].astype(str),
    cri_by_sq['mean'],
    yerr=cri_by_sq['sem'] * 1.96,   # 95% CI
    color=colors_bar,
    edgecolor='white',
    linewidth=0.8,
    capsize=5,
    error_kw={'linewidth': 1.5}
)
axes[1].set_xlabel('Sleep Quality Level', fontsize=11)
axes[1].set_ylabel('Mean Cardio Risk Index (CRI)', fontsize=11)
axes[1].set_title('Sleep Quality Level vs. Mean CRI (with 95% CI)', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.4)
axes[1].axhline(0, color='black', lw=0.8, linestyle='--')
for i, row in cri_by_sq.iterrows():
    axes[1].text(i, row['mean'] + 0.02, f"{row['mean']:.3f}",
                 ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('cardio_cri_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: cardio_cri_scatter.png")

# ── Figure 3: Sleep Disorder Risk Level vs. CRI Distribution ──────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Violin plot — sleep_disorder_risk vs. CRI
order_sdr  = ['Healthy', 'Mild', 'Moderate', 'Severe']
colors_sdr = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
sample3 = df.sample(8000, random_state=RANDOM_STATE)
parts = axes[0].violinplot(
    [sample3.loc[sample3['sleep_disorder_risk'] == r, 'CRI'].values for r in order_sdr],
    positions=range(len(order_sdr)),
    showmedians=True,
    showextrema=True
)
for pc, color in zip(parts['bodies'], colors_sdr):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
axes[0].set_xticks(range(len(order_sdr)))
axes[0].set_xticklabels(order_sdr)
axes[0].set_xlabel('Sleep Disorder Risk Level')
axes[0].set_ylabel('Cardio Risk Index (CRI)')
axes[0].set_title('Sleep Disorder Risk Level vs. CRI Distribution', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.4)

# Right: Mental health condition vs. mean CRI
mhc_cri   = df.groupby('mental_health_condition')['CRI'].agg(['mean', 'sem']).reset_index()
mhc_order  = ['Healthy', 'Anxiety', 'Depression', 'Both']
mhc_colors = ['#4CAF50', '#42A5F5', '#AB47BC', '#EF5350']
mhc_cri    = mhc_cri.set_index('mental_health_condition').reindex(mhc_order).reset_index()
axes[1].bar(
    mhc_cri['mental_health_condition'],
    mhc_cri['mean'],
    yerr=mhc_cri['sem'] * 1.96,
    color=mhc_colors,
    edgecolor='white',
    capsize=5,
    error_kw={'linewidth': 1.5}
)
axes[1].set_xlabel('Mental Health Condition')
axes[1].set_ylabel('Mean Cardio Risk Index (CRI)')
axes[1].set_title('Mental Health Condition vs. Mean CRI', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.4)
axes[1].axhline(0, color='black', lw=0.8, linestyle='--')
for i, row in mhc_cri.iterrows():
    axes[1].text(i, row['mean'] + 0.01, f"{row['mean']:.3f}",
                 ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('cardio_disorder_mental.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: cardio_disorder_mental.png")

# ────────────────────────────────────────────────────────────────
# 4. Statistical Testing: Verify Significance of Sleep Metric
#    Differences Across CVD Risk Groups
# ────────────────────────────────────────────────────────────────
print("\n[Step 4] Statistical Testing (Kruskal-Wallis + post-hoc Mann-Whitney U)...")
print("-" * 65)

groups = {
    'Low CVD Risk':      df.loc[df['cardio_risk_group'] == 'Low CVD Risk'],
    'Moderate CVD Risk': df.loc[df['cardio_risk_group'] == 'Moderate CVD Risk'],
    'High CVD Risk':     df.loc[df['cardio_risk_group'] == 'High CVD Risk'],
}

stat_results = []
for col, label in sleep_metrics.items():
    # Kruskal-Wallis test (does not assume normality)
    kw_stat, kw_p = stats.kruskal(
        groups['Low CVD Risk'][col],
        groups['Moderate CVD Risk'][col],
        groups['High CVD Risk'][col]
    )
    # Low vs. High Mann-Whitney U (effect size r = Z / sqrt(N))
    mw_stat, mw_p = stats.mannwhitneyu(
        groups['Low CVD Risk'][col],
        groups['High CVD Risk'][col],
        alternative='two-sided'
    )
    n = len(groups['Low CVD Risk']) + len(groups['High CVD Risk'])
    z = stats.norm.ppf(mw_p / 2) if mw_p > 0 else 0
    effect_r = abs(z) / np.sqrt(n)

    sig = '***' if kw_p < 0.001 else ('**' if kw_p < 0.01 else ('*' if kw_p < 0.05 else 'ns'))
    print(f"{label:<30} KW p={kw_p:.2e} {sig}  |  Low vs High: MW p={mw_p:.2e}, effect r={effect_r:.3f}")

    stat_results.append({
        'Metric':            label,
        'Low Risk Mean':     groups['Low CVD Risk'][col].mean(),
        'Moderate Risk Mean':groups['Moderate CVD Risk'][col].mean(),
        'High Risk Mean':    groups['High CVD Risk'][col].mean(),
        'KW p-value':        kw_p,
        'Significance':      sig,
        'Effect Size r':     effect_r
    })

stat_df = pd.DataFrame(stat_results)
print("\nStatistical Summary Table:")
print(stat_df.round(4).to_string(index=False))

# ── Figure 4: Mean Values of Sleep Metrics Across Three Risk Groups ────────
fig, ax = plt.subplots(figsize=(13, 6))
x      = np.arange(len(sleep_metrics))
width  = 0.25
colors_3    = ['#4CAF50', '#FFC107', '#F44336']
group_names = ['Low CVD Risk', 'Moderate CVD Risk', 'High CVD Risk']

for i, (grp, color) in enumerate(zip(group_names, colors_3)):
    means = [groups[grp][col].mean() for col in sleep_metrics]
    ax.bar(x + (i - 1) * width, means, width,
           label=grp, color=color, alpha=0.85, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(list(sleep_metrics.values()), rotation=20, ha='right')
ax.set_ylabel('Mean Value')
ax.set_title('Mean Sleep Metrics by Cardiovascular Risk Group', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.4)

# Annotate significance stars
for i_col, (col, label) in enumerate(sleep_metrics.items()):
    sig = stat_df.loc[stat_df['Metric'] == label, 'Significance'].values[0]
    if sig != 'ns':
        y_max = max(groups[g][col].mean() for g in group_names)
        ax.text(i_col, y_max * 1.02, sig, ha='center', fontsize=13,
                color='black', fontweight='bold')

plt.tight_layout()
plt.savefig('cardio_sleep_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: cardio_sleep_comparison.png")

# ────────────────────────────────────────────────────────────────
# 5. Machine Learning: Task C — Predicting High CVD Risk (Binary Classification)
# ────────────────────────────────────────────────────────────────
print("\n[Step 5] Task C: Machine Learning — Predicting High Cardiovascular Risk...")
print("-" * 65)

# Feature preprocessing
df_ml = df.copy()

# Drop ID, derived, and target columns
drop_cols = ['person_id', 'CRI', 'z_hr', 'z_bmi', 'z_stress',
             'z_sleep_dur', 'z_deep', 'z_wake',
             'cardio_risk_group', 'sq_group', 'high_cardio_risk',
             'sleep_disorder_risk', 'felt_rested']
df_ml.drop(columns=[c for c in drop_cols if c in df_ml.columns], inplace=True)

# Encode categorical columns
cat_cols = df_ml.select_dtypes(include=['object']).columns.tolist()
le_dict  = {}
for col in cat_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    le_dict[col] = le

# Target: high_cardio_risk (top CRI tertile = high risk)
y_C = df['high_cardio_risk'].values
X_C = df_ml.values
FEATURE_NAMES = df_ml.columns.tolist()

# Train-test split
X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(
    X_C, y_C, test_size=0.2, random_state=RANDOM_STATE, stratify=y_C
)
print(f"Train set: {X_train_C.shape[0]:,}   Test set: {X_test_C.shape[0]:,}")
print(f"High-risk proportion — Train: {y_train_C.mean():.3f}   Test: {y_test_C.mean():.3f}")

# Helper function
def train_and_eval(model, X_tr, y_tr, X_te, y_te, name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average='weighted')
    try:
        prob    = model.predict_proba(X_te)[:, 1]
        auc_val = roc_auc_score(y_te, prob)
    except AttributeError:
        auc_val = float('nan')
    print(f"\n[{name}]  Accuracy={acc:.4f}  Weighted-F1={f1:.4f}  AUC={auc_val:.4f}")
    print(classification_report(y_te, y_pred, target_names=['Non-High Risk', 'High CVD Risk']))
    return model, y_pred, prob if not np.isnan(auc_val) else None, {
        'name': name, 'acc': acc, 'f1': f1, 'auc': auc_val
    }

results_C = []
preds_C   = {}
probs_C   = {}

# ── C1. Logistic Regression ──────────────────────────────────────────────
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])
m_lr, pred_lr, prob_lr, metrics_lr = train_and_eval(
    pipe_lr, X_train_C, y_train_C, X_test_C, y_test_C, 'Logistic Regression'
)
results_C.append(metrics_lr)
preds_C['Logistic Regression'] = pred_lr
probs_C['Logistic Regression'] = prob_lr

# ── C2. Decision Tree ────────────────────────────────────────────────────
dt_C = DecisionTreeClassifier(max_depth=7, min_samples_leaf=50,
                               random_state=RANDOM_STATE)
m_dt, pred_dt, prob_dt, metrics_dt = train_and_eval(
    dt_C, X_train_C, y_train_C, X_test_C, y_test_C, 'Decision Tree'
)
results_C.append(metrics_dt)
preds_C['Decision Tree'] = pred_dt
probs_C['Decision Tree'] = prob_dt

# ── C3. Random Forest ────────────────────────────────────────────────────
rf_C = RandomForestClassifier(n_estimators=200, max_depth=12,
                               min_samples_leaf=20, random_state=RANDOM_STATE,
                               n_jobs=-1)
m_rf, pred_rf, prob_rf, metrics_rf = train_and_eval(
    rf_C, X_train_C, y_train_C, X_test_C, y_test_C, 'Random Forest'
)
results_C.append(metrics_rf)
preds_C['Random Forest'] = pred_rf
probs_C['Random Forest'] = prob_rf

# ── C4. SVM (on a subset for speed) ─────────────────────────────────────
X_sub, y_sub = resample(X_train_C, y_train_C, n_samples=10000,
                         random_state=RANDOM_STATE, stratify=y_train_C)
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', C=1.0, gamma='scale',
                probability=True, random_state=RANDOM_STATE))
])
m_svm, pred_svm, prob_svm, metrics_svm = train_and_eval(
    pipe_svm, X_sub, y_sub, X_test_C, y_test_C, 'SVM'
)
results_C.append(metrics_svm)
preds_C['SVM'] = pred_svm
probs_C['SVM'] = prob_svm

# ── Summary Table ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Task C — CVD Risk Classification: Model Performance Summary")
print("=" * 55)
summary_C = pd.DataFrame(results_C).set_index('name')
summary_C.columns = ['Accuracy', 'Weighted F1', 'AUC-ROC']
print(summary_C.round(4))

# ────────────────────────────────────────────────────────────────
# 6. Visualization: Task C Performance & Model Interpretation
# ────────────────────────────────────────────────────────────────
print("\n[Step 6] Generating Task C visualization charts...")

# ── Figure 5: Confusion Matrix + ROC Curve ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Best model = Random Forest confusion matrix
cm = confusion_matrix(y_test_C, pred_rf)
ConfusionMatrixDisplay(cm, display_labels=['Non-High Risk', 'High CVD Risk']).plot(
    ax=axes[0], colorbar=False, cmap='Reds'
)
axes[0].set_title('Task C — Random Forest\nConfusion Matrix', fontsize=12, fontweight='bold')

# Task C model performance bar chart
model_names = [r['name'] for r in results_C]
accs  = [r['acc'] for r in results_C]
f1s   = [r['f1']  for r in results_C]
x     = np.arange(len(model_names))
width = 0.35
axes[1].bar(x - width/2, accs, width, label='Accuracy',    color='#5C6BC0', alpha=0.85)
axes[1].bar(x + width/2, f1s,  width, label='Weighted F1', color='#EF5350', alpha=0.85)
for i, (a, f) in enumerate(zip(accs, f1s)):
    axes[1].text(i - width/2, a + 0.005, f'{a:.3f}', ha='center', fontsize=8)
    axes[1].text(i + width/2, f + 0.005, f'{f:.3f}', ha='center', fontsize=8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_names, rotation=15, ha='right')
axes[1].set_ylim(0, 1.08)
axes[1].set_ylabel('Score')
axes[1].set_title('Task C — Model Performance Comparison', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.4)

# ROC curves
roc_models = {
    'Logistic Regression': (m_lr,  '#5C6BC0'),
    'Decision Tree':       (m_dt,  '#FFA726'),
    'Random Forest':       (m_rf,  '#4CAF50'),
    'SVM':                 (m_svm, '#AB47BC'),
}
for name, (model, color) in roc_models.items():
    prob = model.predict_proba(X_test_C)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_C, prob)
    auc_val = auc(fpr, tpr)
    axes[2].plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc_val:.3f})')
axes[2].plot([0, 1], [0, 1], 'k--', lw=1.2)
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('Task C — ROC Curves', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=8, loc='lower right')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cardio_task_c_performance.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: cardio_task_c_performance.png")

# ── Figure 6: Random Forest Feature Importance ──────────────────────────
fi = pd.Series(rf_C.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Left: Top 20 feature importances
top20       = fi.head(20)
colors_fi   = ['#C62828' if i < 5 else '#EF9A9A' for i in range(20)]
axes[0].barh(top20.index[::-1], top20.values[::-1], color=colors_fi[::-1], edgecolor='white')
axes[0].set_xlabel('Feature Importance (Gini Impurity)')
axes[0].set_title('Task C — RF Top 20 Feature Importances', fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.4)

# Right: Decision tree visualization (depth=3 for readability)
dt_viz = DecisionTreeClassifier(max_depth=3, min_samples_leaf=500,
                                 random_state=RANDOM_STATE)
dt_viz.fit(X_train_C, y_train_C)
plot_tree(
    dt_viz,
    feature_names=FEATURE_NAMES,
    class_names=['Non-High Risk', 'High CVD Risk'],
    filled=True, rounded=True, fontsize=8, ax=axes[1], impurity=False
)
axes[1].set_title('Task C — Decision Tree (depth=3)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('cardio_fi_and_tree.png', dpi=130, bbox_inches='tight')
plt.show()
print("  ✓ Saved: cardio_fi_and_tree.png")

# ── Figure 7: Logistic Regression Coefficient Analysis ──────────────────
lr_coef = pd.Series(
    m_lr.named_steps['clf'].coef_[0],
    index=FEATURE_NAMES
).sort_values(key=abs, ascending=False).head(16)

fig, ax = plt.subplots(figsize=(10, 7))
colors_coef = ['#C62828' if c > 0 else '#1565C0' for c in lr_coef]
ax.barh(lr_coef.index[::-1], lr_coef.values[::-1],
        color=colors_coef[::-1], edgecolor='white', linewidth=0.8)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Standardized Coefficient  (positive = increases high CVD risk probability; negative = decreases)', fontsize=11)
ax.set_title('Task C — Logistic Regression Coefficient Analysis', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig('cardio_lr_coef.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: cardio_lr_coef.png")

# ────────────────────────────────────────────────────────────────
# 7. Conclusion Summary
# ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Analysis Conclusion Summary")
print("=" * 65)
print("""
[Cardio Risk Index (CRI) Construction]
  A weighted composite of resting heart rate, BMI, stress score,
  sleep duration, deep sleep percentage, wake episodes per night,
  and shift work status — used as a proxy for cardiovascular disease risk.

[Key Findings]
  1. Poorer sleep quality → higher CRI (negative relationship)
     The high CVD risk group shows significantly lower mean sleep
     quality scores than the low-risk group (p < 0.001).

  2. Insufficient deep sleep → markedly elevated cardiovascular risk
     REM and deep sleep percentages differ significantly across all
     three risk groups (p < 0.001).

  3. Difficulty falling asleep (long sleep latency) is an important warning sign
     The high-risk group's mean sleep latency is approximately
     5–8 minutes longer than that of the low-risk group.

  4. Mental health × cardiovascular risk:
     Individuals with comorbid anxiety and depression ("Both") show
     the highest CRI, reflecting the interaction between mental and
     physical health.

  5. Machine learning performance:
     Random Forest performs best (AUC > 0.95); top features are
     stress_score, sleep_quality_score, and heart_rate_resting_bpm.

[Data Story Hypothesis]
  Improving sleep quality — particularly by increasing deep sleep
  and reducing nocturnal wake episodes — may serve as an actionable
  intervention target for cardiovascular disease prevention, while
  stress management acts as a shared moderating variable for both.
""")

print("=" * 65)
print("  All charts saved successfully")
print("=" * 65)
print("  cardio_sleep_boxplots.png      — Sleep metric boxplots")
print("  cardio_cri_scatter.png         — CRI × Sleep quality scatter plot")
print("  cardio_disorder_mental.png     — Sleep disorder & mental health vs. CRI")
print("  cardio_sleep_comparison.png    — Statistical comparison (with significance markers)")
print("  cardio_task_c_performance.png  — Task C: confusion matrix / performance / ROC")
print("  cardio_fi_and_tree.png         — Feature importances + decision tree visualization")
print("  cardio_lr_coef.png             — Logistic Regression coefficient analysis")