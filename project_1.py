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

