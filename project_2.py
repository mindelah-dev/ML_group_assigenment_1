#project_2
"""
Pump Condition Classification: Normal, Warning, Fault
=====================================================
Chapter 3 Classwork (3.12) - Supervised Machine Learning
Goal     : classify each pump observation as Normal, Warning or Fault.
Features : vibration (mm/s), RPM, temperature (C), pressure (bar), operating hours.
Models   : SVM, Gaussian Naive Bayes, Decision Tree, Random Forest, AdaBoost.
Priority : Fault recall first, then macro F1 and the confusion matrix.

NOTE: the data are synthetic (illustrative only), not real pump measurements.
Predictions should support, not replace, engineering inspection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
CLASS_NAMES = ["Normal", "Warning", "Fault"]
FEATURES = ["Vibration_mm_s", "RPM", "Temperature_C", "Pressure_bar", "Operating_Hours"]
sns.set_theme(style="whitegrid")
============================================================
# PART 1. CREATE THE DATASET (300 records, 100 per condition)
# ============================================================
# For each condition: (mean, standard deviation) of every feature.
PROFILES = {
    "Normal": {
        "Vibration_mm_s": (1.5, 0.45), "RPM": (1480, 35), "Temperature_C": (55, 5),
        "Pressure_bar": (5.0, 0.25), "Operating_Hours": (1800, 700),
    },
   "Warning": {
        "Vibration_mm_s": (4.0, 0.65), "RPM": (1400, 45), "Temperature_C": (72, 6),
        "Pressure_bar": (4.3, 0.30), "Operating_Hours": (4500, 900),
    },
    "Fault": {
        "Vibration_mm_s": (7.0, 0.80), "RPM": (1280, 55), "Temperature_C": (90, 6),
        "Pressure_bar": (3.4, 0.35), "Operating_Hours": (7500, 1000),
    },
}
# Smallest physically sensible value for each feature (avoids negative readings).
MINIMUMS = {
    "Vibration_mm_s": 0.1, "RPM": 500, "Temperature_C": 10,
    "Pressure_bar": 0.1, "Operating_Hours": 0,
}

def create_dataset(records_per_class=100):
    """Generate synthetic pump records for the three conditions."""
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for condition, profile in PROFILES.items():
        for _ in range(records_per_class):
            row = {"Pump_ID": len(rows) + 1}
            for feature, (mean, sd) in profile.items():
                row[feature] = max(MINIMUMS[feature], rng.normal(mean, sd))
            row["Condition"] = condition
            rows.append(row)
 data = pd.DataFrame(rows)
    return data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


df = create_dataset()
print("\nPUMP CONDITION DATASET (first 10 rows)")
print(df.head(10).round(2).to_string(index=False))
print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(df["Condition"].value_counts())

df.to_csv("pump_condition_dataset.csv", index=False)
print("\nSaved dataset to pump_condition_dataset.csv")
# ============================================================
# PART 2. EXPLORE THE DATA
# ============================================================
plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x="Vibration_mm_s", y="Temperature_C",
                hue="Condition", hue_order=CLASS_NAMES, s=65)
plt.title("Pump Vibration vs Temperature by Condition")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Condition", order=CLASS_NAMES)
plt.title("Number of Records in Each Pump Condition")
plt.tight_layout()
plt.show()
plt.figure(figsize=(9, 6))
sns.heatmap(df[FEATURES].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Pump Measurements")
plt.tight_layout()
plt.show()
# ============================================================
# PART 3. SPLIT INTO TRAINING AND TEST DATA
# ============================================================

X = df[FEATURES]
y = df["Condition"]
# stratify=y keeps the same Normal/Warning/Fault ratio in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))

# ============================================================
# PART 4. DEFINE THE FIVE CLASSIFIERS
# ============================================================

# Pipelines fit the scaler on training data only, which prevents data leakage.
models = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ]),
    "Gaussian Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB()),
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    ),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
        n_estimators=50, learning_rate=0.5, random_state=RANDOM_STATE,
    ),
}

# ============================================================
# PART 5. TRAIN AND EVALUATE EACH MODEL
# ============================================================

def fault_score(metric, y_true, y_pred):
    """Return a precision/recall score for the Fault class only."""
    return metric(y_true, y_pred, labels=["Fault"], average=None, zero_division=0)[0]


results = {}

for name, model in models.items():
    print("\n" + "=" * 65)
    print(name.upper())
    print("=" * 65)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Fault Precision": fault_score(precision_score, y_test, y_pred),
        "Fault Recall": fault_score(recall_score, y_test, y_pred),
        "Macro F1": f1_score(y_test, y_pred, labels=CLASS_NAMES,
                             average="macro", zero_division=0),
    }
    for metric, value in results[name].items():
        print(f"{metric:16s}: {value:.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=CLASS_NAMES,
                                target_names=CLASS_NAMES, zero_division=0))
 # 3x3 confusion matrix (rows = actual, columns = predicted).
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ============================================================
# PART 6. COMPARE THE FIVE ALGORITHMS
# ============================================================

results_df = pd.DataFrame(results).T
print("\nMODEL COMPARISON")
print(results_df.round(3))

results_df.plot(kind="bar", figsize=(12, 6), ylim=(0, 1.05))
plt.title("Comparison of Five Pump Condition Classifiers")
plt.ylabel("Evaluation score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ============================================================
# PART 7. FEATURE IMPORTANCE AND THE DECISION TREE
# ============================================================

for name in ["Decision Tree", "Random Forest"]:
    importance = pd.Series(models[name].feature_importances_, index=FEATURES)
    importance = importance.sort_values(ascending=False)

    print(f"\n{name} feature importance:")
    print(importance.round(3))

    plt.figure(figsize=(8, 5))
    importance.sort_values().plot(kind="barh")
    plt.title(f"{name} - Pump Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(20, 10))
plot_tree(models["Decision Tree"], feature_names=FEATURES, class_names=CLASS_NAMES,
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree for Pump Condition Classification")
plt.tight_layout()
plt.show()

# ============================================================
# PART 8. PREDICT A NEW PUMP
# ============================================================

new_pump = pd.DataFrame([{
    "Vibration_mm_s": 5.2, "RPM": 1360, "Temperature_C": 78,
    "Pressure_bar": 4.0, "Operating_Hours": 5200,
}])

print("\nNEW PUMP OBSERVATION")
print(new_pump.to_string(index=False))
print("\nPREDICTIONS FROM THE FIVE ALGORITHMS")
for name, model in models.items():
    print(f"{name:22s}: {model.predict(new_pump)[0]}")




# ============================================================
# PART 9. FINAL SUMMARY
# ============================================================

# Fault recall is the first priority (a missed Fault is the costliest error);
# macro F1 breaks ties.
best_model = results_df.sort_values(["Fault Recall", "Macro F1"], ascending=False).index[0]

print("\n" + "=" * 65)
print("END OF PUMP CONDITION CLASSIFICATION CASE STUDY")
print("=" * 65)
print("Best model in this run (Fault recall, then macro F1):", best_model)
print("These are synthetic classroom data, not a real pump diagnosis.")
print("Predictions should support, not replace, engineering inspection.")
