from sklearn.linear_model import LinearRegression
import numpy as np

# Features: [area, bedrooms]
X = np.array([
    [1000, 2],
    [1200, 3],
    [1500, 3],
    [1800, 4]
])

y = np.array([50, 60, 75, 90])

model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Prediction
print(model.predict([[1600, 3]]))