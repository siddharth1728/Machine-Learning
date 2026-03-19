import numpy as np

# Data
x = np.array([1, 2, 3, 4])
y = np.array([40, 50, 60, 70])

# Mean
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate slope (β1)
num = np.sum((x - x_mean) * (y - y_mean))
den = np.sum((x - x_mean)**2)
b1 = num / den

# Calculate intercept (β0)
b0 = y_mean - b1 * x_mean

print("Intercept:", b0)
print("Slope:", b1)

# Prediction
x_new = 5
y_pred = b0 + b1 * x_new
print("Prediction:", y_pred)