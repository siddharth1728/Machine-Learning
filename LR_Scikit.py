from sklearn.linear_model import LinearRegression
import numpy as np

# Data (2D array required)
X = np.array([[1], [2], [3], [4]])
y = np.array([40, 50, 60, 70])

# Model
model = LinearRegression()

# Train
model.fit(X, y)

# Coefficients
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

# Prediction
y_pred = model.predict([[5]])
print("Prediction:", y_pred)