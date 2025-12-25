Stock Price Prediction using Machine Learning and Flask

This project predicts next-day stock prices using historical stock market data and machine learning techniques. It includes both KNN-based and regression-based models, integrated with a Flask web interface.

It is created for academic learning purposes to understand data preprocessing, model training, prediction workflow, and web deployment.

Technologies Used

Python

Pandas

NumPy

Scikit-learn (KNN & Linear Regression)

Flask

Project Files

app.py – Flask web app to take stock prices input and show predictions

templates/index.html – Input page for stock prices

templates/result.html – Page showing predicted stock price

load.py – Loads and preprocesses the stock dataset

predict.py – Trains the KNN model and predicts stock prices

predict_linear.py – Trains the linear regression model and predicts stock prices

stock_data.csv – Historical stock price dataset

How to Run the Project

Install required libraries:
pip install pandas numpy scikit-learn flask
Keep all files in the same folder.

Run the Flask app:
python app.py

Open the browser at http://127.0.0.1:5000/

Enter today’s stock prices and view the predicted stock for the next day.

Note: The Flask app can be configured to use either predict.py (KNN) or predict_linear.py (Regression).
Objective

Learn machine learning workflows using different models.

Understand stock market data handling and preprocessing.

Perform basic prediction using historical data.

Deploy a simple web interface using Flask.

NOTE: Stock prices depend on many real-world factors.
This project is for educational practice purposes only and not financial advice.
