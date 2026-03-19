from sklearn.tree import DecisionTreeClassifier

# Sample data
X = [[5], [7], [9]]   # CGPA
y = [0, 1, 1]         # 0 = Not placed, 1 = Placed

# Model
model = DecisionTreeClassifier()
model.fit(X, y)

# Prediction
print(model.predict([[6]]))