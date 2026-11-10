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
