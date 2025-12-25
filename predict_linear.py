import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("stock_data.csv")

# -------------------------------
# Features and target
# -------------------------------
X = df[['Stock_1', 'Stock_2', 'Stock_3', 'Stock_4', 'Stock_5']]
y = df['Stock_1'].shift(-1)   # next day's Stock_1

# Drop last row (target becomes NaN)
X = X[:-1]
y = y[:-1]

# -------------------------------
# Split and scale
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Train Linear Regression model
# -------------------------------
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# -------------------------------
# Evaluate model
# -------------------------------
y_pred = lr.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (Linear Regression):", mse)

# -------------------------------
# Interactive prediction
# -------------------------------
print("\nEnter today's stock prices to predict next day's Stock_1:")

try:
    s1 = float(input("Stock_1: "))
    s2 = float(input("Stock_2: "))
    s3 = float(input("Stock_3: "))
    s4 = float(input("Stock_4: "))
    s5 = float(input("Stock_5: "))

    new_data = pd.DataFrame(
        [[s1, s2, s3, s4, s5]],
        columns=['Stock_1', 'Stock_2', 'Stock_3', 'Stock_4', 'Stock_5']
    )

    new_scaled = scaler.transform(new_data)
    prediction = lr.predict(new_scaled)

    print(f"Predicted Stock_1 for next day (Linear Regression): {prediction[0]:.2f}")

except ValueError:
    print("Please enter valid numbers!")
