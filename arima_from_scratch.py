
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------
# Step 1: Load Dataset
# -------------------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url, usecols=[1])
ts = data.values.astype(float).flatten()   # <--- add .flatten()


print("First 10 values of the dataset:")
print(ts[:10])

# Plot original data
plt.plot(ts)
plt.title("Monthly Airline Passengers")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.show()

# -------------------------
# Step 2: Differencing
# -------------------------
def difference(series, d=1):
    diff_series = series.copy()
    for _ in range(d):
        diff_series = diff_series[1:] - diff_series[:-1]
    return diff_series

d = 1
diff_ts = difference(ts, d)
plt.plot(diff_ts)
plt.title("Differenced Series")
plt.show()

# -------------------------
# Step 3: AR model
# -------------------------
def fit_ar(series, p):
    X = np.array([series[i:i+p] for i in range(len(series)-p)])
    y = np.array([series[i+p] for i in range(len(series)-p)])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs

def predict_ar(series, coeffs):
    p = len(coeffs)
    return np.dot(series[-p:], coeffs)

# -------------------------
# Step 4: MA model
# -------------------------
def fit_ma(residuals, q):
    X = np.array([residuals[i:i+q] for i in range(len(residuals)-q)])
    y = np.array([residuals[i+q] for i in range(len(residuals)-q)])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs

def predict_ma(residuals, coeffs):
    q = len(coeffs)
    return np.dot(residuals[-q:], coeffs)

# -------------------------
# Step 5: ARIMA forecast
# -------------------------
def arima_forecast(series, p, d, q, steps=10):
    diff_series = difference(series, d)
    
    # Fit AR
    ar_coeffs = fit_ar(diff_series, p)
    
    # Residuals for MA
    ar_predictions = []
    residuals = []
    for i in range(p, len(diff_series)):
        pred = np.dot(diff_series[i-p:i], ar_coeffs)
        ar_predictions.append(pred)
        residuals.append(diff_series[i] - pred)
    
    # Fit MA
    if q > 0:
        ma_coeffs = fit_ma(np.array(residuals), q)
    else:
        ma_coeffs = np.array([])
    
    # Forecast
    forecast = []
    series_copy = diff_series.copy()
    residuals_copy = residuals.copy()
    
    for _ in range(steps):
        ar_pred = np.dot(series_copy[-p:], ar_coeffs)
        ma_pred = np.dot(residuals_copy[-q:], ma_coeffs) if len(residuals_copy) >= q and q>0 else 0
        next_val = ar_pred + ma_pred
        forecast.append(next_val)
        series_copy = np.append(series_copy, next_val)
        residuals_copy = np.append(residuals_copy, 0)
    
    # Convert back from differencing
    last_val = series[-1]
    for i in range(len(forecast)):
        forecast[i] += last_val
        last_val = forecast[i]
    
    return forecast

# -------------------------
# Step 6: Forecast and plot
# -------------------------
p, d, q = 2, 1, 2
forecast_steps = 12
forecast = arima_forecast(ts, p, d, q, steps=forecast_steps)

plt.plot(ts, label='Actual')
plt.plot(range(len(ts), len(ts)+forecast_steps), forecast, label='Forecast', color='red')
plt.title("ARIMA Forecast")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.legend()
plt.show()

# -------------------------
# Step 7: Evaluation
# -------------------------
# Compare last points only
if forecast_steps <= len(ts):
    mse = mean_squared_error(ts[-forecast_steps:], forecast)
    mae = mean_absolute_error(ts[-forecast_steps:], forecast)
    print("MSE:", mse)
    print("MAE:", mae)
else:
    print("Forecast steps exceed dataset length; skipping evaluation.")
