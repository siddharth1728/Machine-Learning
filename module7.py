"""
Module 7: Decision Trees & Random Forests
==========================================
Full code covering:
  1. Entropy, Gini Index & Information Gain (from scratch)
  2. Decision Tree with sklearn (ID3 / CART)
  3. Pruning (pre-pruning & post-pruning)
  4. Bagging & Bootstrap Aggregation
  5. Random Forests (Bagging + Feature Randomness)
  6. Feature Importance
  7. Use Case: Credit Risk Classification
  8. Evaluation: Accuracy, Confusion Matrix, ROC Curve
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — ENTROPY, GINI & INFORMATION GAIN (From Scratch)
# ═══════════════════════════════════════════════════════════════

def entropy(labels):
    """
    Entropy = -Σ p(x) * log2(p(x))
    Measures impurity/disorder in a set of labels.
    0 = perfectly pure node, 1 = perfectly impure (50/50 split)
    """
    n = len(labels)
    if n == 0:
        return 0
    counts = Counter(labels)
    probs = [count / n for count in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def gini_index(labels):
    """
    Gini = 1 - Σ p(x)^2
    Alternative to entropy. Computationally cheaper.
    0 = pure, 0.5 = max impurity (binary case)
    """
    n = len(labels)
    if n == 0:
        return 0
    counts = Counter(labels)
    probs = [count / n for count in counts.values()]
    return 1 - sum(p ** 2 for p in probs)


def information_gain(parent_labels, left_labels, right_labels, criterion="entropy"):
    """
    Information Gain = Impurity(parent) - weighted avg impurity(children)
    Higher = better split.
    """
    impurity_fn = entropy if criterion == "entropy" else gini_index
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)

    parent_impurity = impurity_fn(parent_labels)
    weighted_child = (n_left / n) * impurity_fn(left_labels) + \
                     (n_right / n) * impurity_fn(right_labels)
    return parent_impurity - weighted_child


# ── Demo ──────────────────────────────────────────────────────
print("=" * 55)
print("SECTION 1: Entropy, Gini & Information Gain")
print("=" * 55)

# Example: weather dataset labels (Play = Yes/No)
labels_parent = ["Yes", "Yes", "Yes", "Yes", "Yes",
                 "No",  "No",  "No",  "No",  "No"]  # 50/50 → max entropy

labels_left  = ["Yes", "Yes", "Yes", "Yes"]          # all Yes → 0 entropy
labels_right = ["Yes", "No",  "No",  "No",  "No",  "No"]  # mostly No

print(f"\nParent entropy  : {entropy(labels_parent):.4f}  (expected ~1.0)")
print(f"Pure node entropy: {entropy(labels_left):.4f}   (expected 0.0)")
print(f"Parent gini     : {gini_index(labels_parent):.4f}  (expected 0.5)")
print(f"Pure node gini  : {gini_index(labels_left):.4f}   (expected 0.0)")

ig = information_gain(labels_parent, labels_left, labels_right)
print(f"\nInformation Gain (entropy): {ig:.4f}")
ig_gini = information_gain(labels_parent, labels_left, labels_right, "gini")
print(f"Information Gain (gini)   : {ig_gini:.4f}")


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — DECISION TREE (sklearn — CART algorithm)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("SECTION 2: Decision Tree with sklearn")
print("=" * 55)

# ── Toy dataset: weather → play cricket? ─────────────────────
weather_data = {
    "Outlook":    ["Sunny","Sunny","Overcast","Rain","Rain","Rain",
                   "Overcast","Sunny","Sunny","Rain","Sunny","Overcast",
                   "Overcast","Rain"],
    "Temperature":["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild",
                   "Cool","Mild","Mild","Mild","Hot","Mild"],
    "Humidity":   ["High","High","High","High","Normal","Normal","Normal",
                   "High","Normal","Normal","Normal","High","Normal","High"],
    "Wind":       ["Weak","Strong","Weak","Weak","Weak","Strong","Strong",
                   "Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
    "PlayCricket":["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes",
                   "Yes","Yes","Yes","No"],
}
df_weather = pd.DataFrame(weather_data)
print("\nWeather Dataset:")
print(df_weather.to_string(index=False))

# Encode categorical features
le = LabelEncoder()
df_encoded = df_weather.copy()
for col in df_encoded.columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop("PlayCricket", axis=1)
y = df_encoded["PlayCricket"]
feature_names = X.columns.tolist()

# Full tree (no pruning)
dt_full = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt_full.fit(X, y)
print("\n--- Full Decision Tree (ID3 / entropy) ---")
print(export_text(dt_full, feature_names=feature_names))

# Visualize tree
fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(
    dt_full,
    feature_names=feature_names,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    ax=ax,
    fontsize=9,
)
ax.set_title("Full Decision Tree — Play Cricket Dataset", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("decision_tree_full.png", dpi=120)
plt.close()
print("Saved: decision_tree_full.png")


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — PRUNING
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("SECTION 3: Pruning — Pre & Post")
print("=" * 55)

# We use the Credit Risk dataset (built below) for a clearer demo.
# Synthetic credit risk data
np.random.seed(42)
n = 500
credit_data = pd.DataFrame({
    "Age":            np.random.randint(20, 70, n),
    "Income":         np.random.randint(20000, 120000, n),
    "LoanAmount":     np.random.randint(5000, 50000, n),
    "CreditHistory":  np.random.choice([0, 1], n, p=[0.3, 0.7]),  # 0=bad, 1=good
    "Employment":     np.random.choice([0, 1, 2], n),              # 0=unemployed,1=part,2=full
    "DebtRatio":      np.round(np.random.uniform(0.1, 0.9, n), 2),
})
# Simple rule: good credit + decent income → low risk
credit_data["Risk"] = ((credit_data["CreditHistory"] == 1) &
                       (credit_data["Income"] > 50000) &
                       (credit_data["DebtRatio"] < 0.5)).astype(int)

X_cr = credit_data.drop("Risk", axis=1)
y_cr = credit_data["Risk"]
X_train, X_test, y_train, y_test = train_test_split(
    X_cr, y_cr, test_size=0.2, random_state=42
)

# ── Pre-pruning: limit depth + min_samples_split ──────────────
dt_prepruned = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,              # pre-pruning: stop at depth 4
    min_samples_split=20,     # pre-pruning: need ≥20 samples to split
    min_samples_leaf=10,      # pre-pruning: leaf needs ≥10 samples
    random_state=42,
)
dt_prepruned.fit(X_train, y_train)
acc_pre = accuracy_score(y_test, dt_prepruned.predict(X_test))
print(f"\nPre-pruned tree  | max_depth=4 | Accuracy: {acc_pre:.4f}")

# ── Post-pruning: Cost Complexity Pruning (CCP) ───────────────
# Find optimal alpha (ccp_alpha) via cross-validation
path = DecisionTreeClassifier(random_state=42).fit(X_train, y_train).cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # remove last (trivial empty tree)

cv_scores = []
for alpha in ccp_alphas:
    dt_tmp = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(dt_tmp, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

best_alpha = ccp_alphas[np.argmax(cv_scores)]
dt_postpruned = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
dt_postpruned.fit(X_train, y_train)
acc_post = accuracy_score(y_test, dt_postpruned.predict(X_test))
print(f"Post-pruned tree | best ccp_alpha={best_alpha:.4f} | Accuracy: {acc_post:.4f}")

# ── Compare: unpruned vs pruned ───────────────────────────────
dt_unpruned = DecisionTreeClassifier(criterion="gini", random_state=42)
dt_unpruned.fit(X_train, y_train)
acc_unp = accuracy_score(y_test, dt_unpruned.predict(X_test))
print(f"Unpruned tree    | depth={dt_unpruned.get_depth()} | Accuracy: {acc_unp:.4f}")

# Plot CCP alpha vs accuracy
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ccp_alphas, cv_scores, marker="o", markersize=4, color="#028090", linewidth=1.5)
ax.axvline(best_alpha, color="#F96167", linestyle="--", label=f"Best α = {best_alpha:.4f}")
ax.set_xlabel("ccp_alpha (pruning strength)")
ax.set_ylabel("CV Accuracy")
ax.set_title("Post-Pruning: Accuracy vs CCP Alpha", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("pruning_ccp_alpha.png", dpi=120)
plt.close()
print("Saved: pruning_ccp_alpha.png")


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — BAGGING (Bootstrap Aggregation)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("SECTION 4: Bagging — Bootstrap Aggregation")
print("=" * 55)

# ── Bootstrap sampling demo ───────────────────────────────────
def bootstrap_sample(X, y, seed=None):
    """Returns a sample of size n drawn WITH replacement."""
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.integers(0, n, size=n)    # sample WITH replacement
    return X.iloc[indices], y.iloc[indices]

print("\nBootstrap sample demo (first 5 indices):")
for i in range(3):
    Xb, yb = bootstrap_sample(X_train, y_train, seed=i)
    unique_count = len(set(Xb.index.tolist()))
    pct_unique = unique_count / len(X_train) * 100
    print(f"  Sample {i+1}: {unique_count}/{len(X_train)} unique rows ({pct_unique:.1f}%) — ~63% expected")

# sklearn BaggingClassifier
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=50,
    max_samples=1.0,    # 100% rows per sample (with replacement)
    max_features=1.0,   # 100% features per sample
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)
bagging_clf.fit(X_train, y_train)
acc_bag = accuracy_score(y_test, bagging_clf.predict(X_test))
print(f"\nBagging (50 trees, all features) Accuracy: {acc_bag:.4f}")


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — RANDOM FORESTS
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("SECTION 5: Random Forests")
print("=" * 55)

# ── Bagging vs Feature Bagging (Random Subspace) ─────────────
# Feature Bagging: each split considers only sqrt(n_features) features
rf_clf = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,          # let trees grow fully (forest handles overfitting)
    max_features="sqrt",     # Random Subspace Method: sqrt(n_features) per split
    bootstrap=True,          # Bagging: sample with replacement
    oob_score=True,          # Out-of-Bag error estimate
    random_state=42,
    n_jobs=-1,
)
rf_clf.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, rf_clf.predict(X_test))
print(f"\nRandom Forest (100 trees):")
print(f"  Test Accuracy  : {acc_rf:.4f}")
print(f"  OOB Score      : {rf_clf.oob_score_:.4f}  ← free validation without splitting data")

# ── Effect of number of trees ─────────────────────────────────
n_tree_list = [1, 5, 10, 20, 50, 100, 200]
train_accs, test_accs = [], []
for n in n_tree_list:
    rf_tmp = RandomForestClassifier(n_estimators=n, max_features="sqrt",
                                    random_state=42, n_jobs=-1)
    rf_tmp.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, rf_tmp.predict(X_train)))
    test_accs.append(accuracy_score(y_test, rf_tmp.predict(X_test)))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(n_tree_list, train_accs, marker="o", label="Train", color="#028090", linewidth=1.5)
ax.plot(n_tree_list, test_accs,  marker="s", label="Test",  color="#F96167", linewidth=1.5)
ax.set_xlabel("Number of Trees (n_estimators)")
ax.set_ylabel("Accuracy")
ax.set_title("Random Forest: Effect of Number of Trees", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rf_n_estimators.png", dpi=120)
plt.close()
print("Saved: rf_n_estimators.png")


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("SECTION 6: Feature Importance")
print("=" * 55)

importances = rf_clf.feature_importances_
feat_names  = X_cr.columns.tolist()
feat_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=False)

print("\nFeature Importances (Random Forest):")
print(feat_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#028090" if i == 0 else "#97BC62" for i in range(len(feat_df))]
ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors, edgecolor="none")
ax.invert_yaxis()
ax.set_xlabel("Mean Decrease in Gini Impurity")
ax.set_title("Feature Importance — Credit Risk (Random Forest)", fontweight="bold")
for i, (val, name) in enumerate(zip(feat_df["Importance"], feat_df["Feature"])):
    ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.close()
print("Saved: feature_importance.png")


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — USE CASE: CREDIT RISK CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("SECTION 7: Use Case — Credit Risk Classification")
print("=" * 55)

print("\nDataset shape:", credit_data.shape)
print("Class balance:\n", credit_data["Risk"].value_counts())
print("\nSample rows:")
print(credit_data.head(6).to_string(index=False))

# ── Train three models for comparison ────────────────────────
models = {
    "Decision Tree (unpruned)": DecisionTreeClassifier(random_state=42),
    "Decision Tree (pruned)":   DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest":            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    results[name] = {"model": model, "y_pred": y_pred, "y_prob": y_prob,
                     "acc": acc, "fpr": fpr, "tpr": tpr, "auc": roc_auc}
    print(f"\n{name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=["High Risk", "Low Risk"], zero_division=0))


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — EVALUATION: CONFUSION MATRIX & ROC CURVE
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("SECTION 8: Evaluation — Confusion Matrix & ROC Curve")
print("=" * 55)

# ── 1. Confusion Matrices (all 3 models side by side) ─────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["High Risk", "Low Risk"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nAcc={res['acc']:.3f}", fontsize=10, fontweight="bold")
plt.suptitle("Confusion Matrices — Credit Risk", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: confusion_matrices.png")

# ── 2. ROC Curves ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
colors_roc = ["#028090", "#F96167", "#2C5F2D"]
for (name, res), color in zip(results.items(), colors_roc):
    ax.plot(res["fpr"], res["tpr"], label=f"{name} (AUC={res['auc']:.3f})",
            color=color, linewidth=2)
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Credit Risk Classification", fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=120)
plt.close()
print("Saved: roc_curves.png")

# ── 3. Summary comparison bar chart ───────────────────────────
model_names = list(results.keys())
accuracies  = [results[n]["acc"] for n in model_names]
aucs        = [results[n]["auc"] for n in model_names]

x = np.arange(len(model_names))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 4))
bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy", color="#028090", alpha=0.85)
bars2 = ax.bar(x + width/2, aucs,       width, label="ROC-AUC",  color="#97BC62", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(" (", "\n(") for n in model_names], fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Credit Risk", fontweight="bold")
ax.legend()
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120)
plt.close()
print("Saved: model_comparison.png")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print("""
Concept               | Covered in Section
─────────────────────────────────────────
Entropy               | Section 1
Gini Index            | Section 1
Information Gain      | Section 1
Decision Tree (CART)  | Section 2
Pruning (pre/post)    | Section 3
Bootstrap Bagging     | Section 4
Random Forest         | Section 5
Feature Importance    | Section 6
Credit Risk Use Case  | Section 7
Confusion Matrix      | Section 8
ROC Curve             | Section 8
Model Comparison      | Section 8

Plots saved:
  decision_tree_full.png   — Full unpruned decision tree
  pruning_ccp_alpha.png    — Post-pruning alpha vs accuracy
  rf_n_estimators.png      — Effect of number of trees
  feature_importance.png   — Feature importances (RF)
  confusion_matrices.png   — Side-by-side confusion matrices
  roc_curves.png           — ROC curves for all 3 models
  model_comparison.png     — Accuracy & AUC comparison bar chart
""")