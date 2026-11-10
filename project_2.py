#project_2"""
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
