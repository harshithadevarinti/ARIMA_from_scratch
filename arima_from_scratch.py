import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
# 1. Load Dataset (Apple Stock Prices)

url = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
data = pd.read_csv(url)

# Use the 'AAPL.Close' column
ts = data['AAPL.Close'].values.astype(float).flatten()

print("First 10 values of the dataset:")
print(ts[:10])

# 2. Differencing Function (I part)
def difference(series, interval=1):
    diff = []
    for i in range(interval, len(series)):
        diff.append(series[i] - series[i - interval])
    return np.array(diff)

# 3. Autoregression (AR part)

def fit_ar(series, p):
    X, y = [], []
    for i in range(p, len(series)):
        X.append(series[i - p:i])
        y.append(series[i])
    X, y = np.array(X), np.array(y)
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs

def predict_ar(history, coeffs):
    return np.dot(coeffs, history[-len(coeffs):])

# 4. Moving Average (MA part)

def predict_ma(errors, q):
    if len(errors) < q:
        q = len(errors)
    return np.mean(errors[-q:]) if q > 0 else 0

# 5. Full ARIMA Model

def arima_forecast(series, p, d, q, steps=10):
    history = list(series)
    errors = []
    predictions = []

    # Apply differencing
    diff_series = difference(history, d)
    ar_coeffs = fit_ar(diff_series, p)

    for t in range(steps):
        diff_history = difference(history, d)
        ar_part = predict_ar(diff_history, ar_coeffs) if len(diff_history) >= p else 0
        ma_part = predict_ma(errors, q)
        yhat = ar_part + ma_part

        # Inverse differencing
        forecast = history[-d] + yhat
        predictions.append(forecast)

        # Append forecast to history
        history.append(forecast)

        # Compute error
        if len(history) > len(series):
            error = series[len(history) - 1] - forecast if (len(series) > len(history) - 1) else 0
            errors.append(error)

    return predictions

# 6. Run Model

p, d, q = 2, 1, 2   # ARIMA parameters
forecast_steps = 20

forecast = arima_forecast(ts, p, d, q, steps=forecast_steps)

# 7. Evaluation

test = ts[-forecast_steps:]
pred = forecast[:len(test)]

mse = mean_squared_error(test, pred)
mae = mean_absolute_error(test, pred)

print("MSE:", mse)
print("MAE:", mae)

# 8. Visualization

plt.figure(figsize=(10,5))
plt.plot(ts, label='Actual Data', color='blue')
plt.plot(range(len(ts), len(ts) + forecast_steps), forecast, label='Forecast', color='red')
plt.legend()
plt.title("ARIMA Forecast on Apple Stock Prices")
plt.show()
print("this is for demonstrating branches")
