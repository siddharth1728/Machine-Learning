from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4]]
y = [3, 5, 7, 9]

model = LinearRegression()
model.fit(X, y)

print(model.predict([[5]]))  # Output: ~11