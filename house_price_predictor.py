import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample dataset: square footage vs. house price

# Step-1 Identify the Data
data = {
    'SquareFeet': [600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    'Price': [150000, 180000, 200000, 220000, 260000, 300000, 320000, 360000]
}

# Step-2 Split the data into X and Y values
df = pd.DataFrame(data)
X = df[['SquareFeet']]   # Features
Y = df["Price"]         # Target

# Step-3 Split Data into Test Set and Training Set
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.25, random_state=42
)
# test_size = 0.25 means 25% of the data is in the test set (25 out of 100 samples will be test data)

# Step-4 Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step-5 Make Predictions
y_pred = model.predict(X_test)

# Step-6 Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Step-7 Visualize the Results
plt.scatter(X, Y, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('House Price Prediction')
plt.legend()
plt.show()




