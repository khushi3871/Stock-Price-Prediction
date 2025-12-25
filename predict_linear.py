import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("stock_data.csv")

# -------------------------------
# Create lag features
# -------------------------------
df['Stock_1_lag1'] = df['Stock_1'].shift(1)
df['Stock_1_lag2'] = df['Stock_1'].shift(2)
df = df.dropna()

# -------------------------------
# Features and target
# -------------------------------
X = df[['Stock_1_lag1', 'Stock_1_lag2', 'Stock_2', 'Stock_3', 'Stock_4', 'Stock_5']]
y = df['Stock_1']

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
# Hyperparameter tuning with Ridge
# -------------------------------
ridge = Ridge()
params = {'alpha': [0.01, 0.1, 1, 10, 50, 100]}
grid = GridSearchCV(ridge, param_grid=params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

print("Best alpha:", grid.best_params_)
print("Mean Squared Error (Ridge):", mse)

# -------------------------------
# Interactive prediction
# -------------------------------
print("\nEnter today's stock prices:")
try:
    s1_lag1 = float(input("Stock_1 yesterday: "))
    s1_lag2 = float(input("Stock_1 day before yesterday: "))
    s2 = float(input("Stock_2 today: "))
    s3 = float(input("Stock_3 today: "))
    s4 = float(input("Stock_4 today: "))
    s5 = float(input("Stock_5 today: "))

    new_data = pd.DataFrame(
        [[s1_lag1, s1_lag2, s2, s3, s4, s5]],
        columns=['Stock_1_lag1','Stock_1_lag2','Stock_2','Stock_3','Stock_4','Stock_5']
    )

    new_scaled = scaler.transform(new_data)
    prediction = best_model.predict(new_scaled)

    print(f"\nPredicted Stock_1 for next day: {prediction[0]:.2f}")

except ValueError:
    print("Please enter valid numbers!")
