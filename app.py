from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# -------------------------------
# Load dataset and train model once
# -------------------------------
df = pd.read_csv("stock_data.csv")
df['Stock_1_lag1'] = df['Stock_1'].shift(1)
df['Stock_1_lag2'] = df['Stock_1'].shift(2)
df = df.dropna()

X = df[['Stock_1_lag1', 'Stock_1_lag2', 'Stock_2', 'Stock_3', 'Stock_4', 'Stock_5']]
y = df['Stock_1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
# Flask routes
# -------------------------------

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        s1_lag1 = float(request.form['s1_lag1'])
        s1_lag2 = float(request.form['s1_lag2'])
        s2 = float(request.form['s2'])
        s3 = float(request.form['s3'])
        s4 = float(request.form['s4'])
        s5 = float(request.form['s5'])

        new_data = pd.DataFrame([[s1_lag1, s1_lag2, s2, s3, s4, s5]],
                                columns=['Stock_1_lag1','Stock_1_lag2','Stock_2','Stock_3','Stock_4','Stock_5'])

        new_scaled = scaler.transform(new_data)
        prediction = best_model.predict(new_scaled)[0]

        # Render a separate result page
        return render_template('result.html', prediction=round(prediction, 2))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
