import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "distance_km": [5, 10, 3, 8, 15],
    "avg_speed_kmph": [30, 40, 25, 35, 45],
    "car_size": [1, 2, 1, 2, 3],
    "trip_type": [0, 1, 0, 1, 0],
    "driver_experience": [2, 5, 1, 4, 6],
    "trip_duration_min": [12, 18, 10, 20, 25]
}

df = pd.DataFrame(data)

X = df.drop("trip_duration_min", axis=1)
y = df["trip_duration_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Predicted:", prediction)
print("Actual:", y_test.values)

new_trip = [[7, 35, 2, 1, 4]]

predicted_time = model.predict(new_trip)

print("Predicted Trip Duration:", predicted_time[0])