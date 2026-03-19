from sklearn.linear_model import LogisticRegression  # Import classification model

X = [[10], [20], [30], [40]]  
# Input feature → number of "suspicious words" in email (e.g., free, win, offer)

y = [0, 0, 1, 1]  
# Output labels:
# 0 = Not Spam
# 1 = Spam

model = LogisticRegression()  # Create Logistic Regression model

model.fit(X, y)  # Train the model → learns pattern between X and y

prediction = model.predict([[25]])  
# Predict for new email with 25 suspicious words

print(prediction)  
# Output → [1] means Spam, [0] means Not Spam

print(model.coef_)  
# Shows importance of feature (how strongly words affect spam detection)

print(model.intercept_)  
# Base value used in decision boundary



# 