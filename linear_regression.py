# METHOD OF SUPERVISED LEARNING


# 1. Linear Regression 


from sklearn.linear_model import LinearRegression  # Import Linear Regression model from sklearn

X = [[1], [2], [3], [4]]  # Input feature (2D list required by sklearn) → years of experience

y = [3, 5, 7, 9]  # Target values → salary corresponding to each experience

model = LinearRegression()  # Create an instance of the Linear Regression model

model.fit(X, y)  # Train the model → finds the best-fit line (learns m and b)

# m → Slope (growth rate)
# b → Intercept (starting value)

prediction = model.predict([[5]])  # Predict salary for 5 years of experience

print(prediction)  # Output the predicted value (expected ≈ 11)

print(model.coef_)      # Print slope (m) → how much salary increases per year

print(model.intercept_) # Print intercept (b) → base salary when experience = 0