"""
PRACTICAL ENGINEERING CASE STUDY
Supervised Machine Learning Classification for Machine Failure Risk


Algorithms:
1. Support Vector Machine (SVM)
2. Gaussian Naive Bayes
3. Decision Tree
4. Random Forest
5. AdaBoost


IMPORTANT:
- This project uses a small, invented dataset for learning and demonstration.
- The values are NOT real industrial measurements.
- The predictions are NOT a real machine safety or maintenance diagnosis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

#============================================================
# PART 1. CREATE A NEW DATASET (30 MACHINE RECORDS)
# ============================================================


# Columns:
# Machine_ID, Temperature_C, Vibration_mm_s,
# Speed_RPM, Operating_Hours, Failure_Risk
#
# Failure_Risk:
# 0 = Normal
# 1 = Failure risk

data = [
    [1,  56, 1.8, 1460, 1200, 0],
    [2,  61, 2.1, 1445, 1800, 0],
    [3,  65, 2.5, 1420, 2400, 0],
    [4,  59, 1.9, 1470, 1500, 0],
    [5,  68, 2.8, 1410, 3000, 0],
    [6,  63, 2.3, 1450, 2200, 0],
    [7,  71, 3.2, 1395, 3600, 0],
    [8,  54, 1.6, 1480, 1000, 0],
    [9,  66, 2.6, 1430, 2800, 0],
    [10, 72, 3.5, 1380, 4000, 0],
    [11, 58, 2.0, 1465, 1600, 0],
    [12, 69, 3.0, 1400, 3400, 0],
    [13, 62, 2.4, 1440, 2100, 0],
    [14, 74, 3.7, 1375, 4500, 0],
    [15, 67, 2.7, 1425, 2900, 0],


    [16, 77, 4.3, 1360, 4900, 1],
    [17, 81, 4.8, 1340, 5300, 1],
    [18, 85, 5.2, 1320, 5900, 1],
    [19, 79, 4.5, 1350, 5100, 1],
    [20, 88, 5.8, 1300, 6500, 1],
    [21, 83, 5.0, 1330, 5700, 1],
    [22, 91, 6.2, 1280, 7100, 1],
    [23, 76, 4.1, 1370, 4700, 1],
    [24, 86, 5.5, 1310, 6200, 1],
    [25, 80, 4.7, 1345, 5500, 1],
    [26, 94, 6.7, 1260, 7800, 1],
    [27, 78, 4.4, 1365, 5000, 1],
    [28, 89, 6.0, 1290, 6800, 1],
    [29, 84, 5.3, 1325, 6000, 1],
    [30, 92, 6.5, 1270, 7500, 1],
]
columns = [
    "Machine_ID",
    "Temperature_C",
    "Vibration_mm_s",
    "Speed_RPM",
    "Operating_Hours",
    "Failure_Risk",
]


df = pd.DataFrame(data, columns=columns)


print("\n" + "=" * 65)
print("DATASET: 30 MACHINE RECORDS")
print("=" * 65)
print(df.to_string(index=False))


print("\nCLASS DISTRIBUTION")
print(
    df["Failure_Risk"]
    .value_counts()
    .sort_index()
    .rename(index={0: "Normal", 1: "Failure risk"})
)


# Separate input features (X) from the answer/target (y).
feature_cols = [
    "Temperature_C",
    "Vibration_mm_s",
    "Speed_RPM",
    "Operating_Hours",
]
X = df[feature_cols]
y = df["Failure_Risk"]



# ============================================================
# PART 2. VISUALIZE THE DATASET
# ============================================================



sns.set_theme(style="whitegrid")



# Scatter plot: temperature compared with vibration.
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Temperature_C",
    y="Vibration_mm_s",
    hue="Failure_Risk",
    style="Failure_Risk",
    s=100,
)


plt.title("Machine Temperature vs. Vibration")
plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration (mm/s)")
plt.legend(title="0 = Normal, 1 = Failure risk")
plt.tight_layout()
plt.show()



# Correlation heatmap.
plt.figure(figsize=(8, 5))
sns.heatmap(
    df[feature_cols + ["Failure_Risk"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
)
plt.title("Correlation Matrix of Engineering Variables")
plt.tight_layout()
plt.show()



# ============================================================
# PART 3. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================
# 70% is used for training and 30% for testing.
# Stratification keeps the class ratio similar in both groups.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)


print("\nTRAINING AND TESTING DATA")
print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")


# ============================================================
# PART 4. DEFINE FIVE MACHINE LEARNING CLASSIFIERS
# ============================================================


# SVM: scaling is important because SVM uses distances.
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale")),
])


# Gaussian Naive Bayes.
naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB()),
])


# Decision Tree.
decision_tree_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42,
)
# Random Forest.
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42,
)
# AdaBoost.
# The estimator argument is used by modern versions of scikit-learn.
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42,
    ),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42,
)
models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model,
}
# ============================================================
# PART 5. TRAIN, PREDICT, EVALUATE, AND VISUALIZE EACH MODEL
# ============================================================

results = []
predictions = {}

for model_name, model in models.items():
 # Predict the labels for the testing data.
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    print("\n" + "=" * 65)
    print(model_name.upper())
    print("=" * 65)
 # Train the model using the training data.
    model.fit(X_train, y_train)
 # Predict the labels for the testing data.
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred
 # Calculate evaluation measurements.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
 results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    })
 print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
 print("\nCLASSIFICATION REPORT")
    print(
  y_pred,
            labels=[0, 1],
            target_names=["Normal", "Failure risk"],
            zero_division=0,
        )
    )
 # Draw a confusion matrix for this model.
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
   labels=[0, 1],
        display_labels=["Normal", "Failure risk"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    ) 
 ax.set_title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.show()
# ============================================================
# PART 6. COMPARE THE FIVE ALGORITHMS
# ============================================================
results_df = pd.DataFrame(results).set_index("Algorithm")

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)
print(results_df.round(3))
# Compare accuracy, precision, recall, and F1-score.
results_df.plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1.05),
)
plt.title("Comparison of Five Supervised Learning Algorithms")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
 ============================================================
# PART 7. FEATURE IMPORTANCE AND DECISION TREE
# ============================================================

tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
}

for name, model in tree_models.items():

    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols,
    ).sort_values(ascending=False)
 print(f"\n{name} - FEATURE IMPORTANCE")
    print(importance.round(3))

    plt.figure(figsize=(7, 4))
    plt.barh(importance.index, importance.values)
    plt.title(f"{name}: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Engineering Feature")
    plt.tight_layout()
    plt.show()

# Draw the trained Decision Tree.
plt.figure(figsize=(16, 8))
plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=["Normal", "Failure risk"],
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.title("Decision Tree Learned from the Training Records")
plt.tight_layout()
plt.show()
 ============================================================
# PART 8. TEST A NEW MACHINE OBSERVATION
# ============================================================
 This is a new example, not one of the original 30 records.
# Its values are invented for this educational demonstration.
new_machine = pd.DataFrame([{
    "Temperature_C": 75,
    "Vibration_mm_s": 3.9,
    "Speed_RPM": 1380,
    "Operating_Hours": 4600,
}])
print("\n" + "=" * 65)
print("NEW MACHINE OBSERVATION")
print("=" * 65)
print(new_machine.to_string(index=False))
print("\nPREDICTIONS FROM THE FIVE ALGORITHMS")

for name, model in models.items():
    predicted_class = int(model.predict(new_machine)[0])

    if predicted_class == 1:
        label = "Failure risk"
    else:
        label = "Normal"
 print(f"{name:15s} prediction: {label}")
