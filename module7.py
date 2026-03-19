# ==============================
# 1. Import Libraries
# ==============================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
# 2. Create Sample Dataset
# (Placement Example - Real Life)
# ==============================
data = {
    'CGPA': [5, 6, 7, 8, 9, 4, 6, 7, 8, 9],
    'Skills': [2, 3, 5, 7, 8, 1, 4, 6, 7, 9],
    'Projects': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'Placed': [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# ==============================
# 3. Split Data
# ==============================
X = df[['CGPA', 'Skills', 'Projects']]
y = df['Placed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==============================
# 4. Decision Tree Model
# ==============================
dt_model = DecisionTreeClassifier(
    criterion='gini',   # can also use 'entropy'
    max_depth=3         # helps in pruning
)

dt_model.fit(X_train, y_train)

# Prediction
dt_pred = dt_model.predict(X_test)

# Evaluation
print("=== Decision Tree ===")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("Report:\n", classification_report(y_test, dt_pred))

# ==============================
# 5. Random Forest Model
# ==============================
rf_model = RandomForestClassifier(
    n_estimators=10,    # number of trees
    max_depth=3,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Prediction
rf_pred = rf_model.predict(X_test)

# Evaluation
print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Report:\n", classification_report(y_test, rf_pred))

# ==============================
# 6. Feature Importance
# ==============================
importance = rf_model.feature_importances_

feature_names = X.columns

print("\nFeature Importance:")
for i in range(len(feature_names)):
    print(f"{feature_names[i]}: {importance[i]:.3f}")

# ==============================
# 7. Predict New Student
# ==============================
new_student = [[7.5, 6, 3]]  # CGPA, Skills, Projects

dt_result = dt_model.predict(new_student)
rf_result = rf_model.predict(new_student)

print("\nNew Student Prediction:")
print("Decision Tree:", "Placed" if dt_result[0] == 1 else "Not Placed")
print("Random Forest:", "Placed" if rf_result[0] == 1 else "Not Placed")