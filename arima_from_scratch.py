import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------------------------------------------------
# 1. Load Dataset (Apple Stock Prices)
# ------------------------------------------------------------------
url = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
data = pd.read_csv(url)
ts = data['AAPL.Close'].values.astype(float).flatten()
print("First 10 values of the dataset:")
print(ts[:10])


# ------------------------------------------------------------------
# 2. Differencing (I part) — general order d, with inverse support
# ------------------------------------------------------------------
def difference(series, d=1):
    """Apply differencing d times."""
    diff = np.array(series, dtype=float)
    for _ in range(d):
        diff = np.diff(diff)
    return diff


def inverse_difference(history, forecast_diff, d=1):
    """
    Undo d-th order differencing for a single forecasted value.
    Uses the last d+1 values of `history` (original scale) to
    reconstruct the next value from a differenced forecast.
    """
    # Recursively integrate d times
    vals = list(history)
    stack = []
    series = np.array(vals, dtype=float)
    for _ in range(d - 1):
        series = np.diff(series)
        stack.append(series)
    # start from the innermost difference level and add forecast_diff back
    value = forecast_diff
    for level in reversed(stack):
        value = value + level[-1]
    value = value + vals[-1] if d >= 1 else value
    return value


# ------------------------------------------------------------------
# 3. Autoregression (AR part) — with intercept term
# ------------------------------------------------------------------
def fit_ar(series, p):
    """Fit AR(p) coefficients (+ intercept) via least squares."""
    X, y = [], []
    for i in range(p, len(series)):
        X.append(series[i - p:i])
        y.append(series[i])
    X, y = np.array(X), np.array(y)
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # add intercept column
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs  # coeffs[0] = intercept, coeffs[1:] = AR weights


def predict_ar(history, coeffs):
    lag_vals = np.array(history[-(len(coeffs) - 1):])
    return coeffs[0] + np.dot(coeffs[1:], lag_vals)


# ------------------------------------------------------------------
# 4. Moving Average (MA part) — actually fit MA weights on residuals
#    via least squares, not just a running mean.
# ------------------------------------------------------------------
def fit_ma(residuals, q):
    """Fit MA(q) coefficients on the AR residual series."""
    if len(residuals) <= q:
        return np.zeros(q + 1)
    X, y = [], []
    for i in range(q, len(residuals)):
        X.append(residuals[i - q:i])
        y.append(residuals[i])
    X, y = np.array(X), np.array(y)
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs  # coeffs[0] = intercept, coeffs[1:] = MA weights


def predict_ma(errors, coeffs, q):
    if len(errors) < q or q == 0:
        return coeffs[0] if len(coeffs) else 0.0
    lag_errs = np.array(errors[-q:])
    return coeffs[0] + np.dot(coeffs[1:], lag_errs)


# ------------------------------------------------------------------
# 5. Full ARIMA Model
# ------------------------------------------------------------------
def arima_forecast(series, p, d, q, steps=10):
    history = list(series)

    # --- Fit AR on differenced training series ---
    diff_series = difference(history, d)
    ar_coeffs = fit_ar(diff_series, p)

    # --- Build in-sample AR residuals, then fit MA on them ---
    ar_fitted = []
    for i in range(p, len(diff_series)):
        pred = predict_ar(diff_series[:i], ar_coeffs)
        ar_fitted.append(pred)
    ar_actual = diff_series[p:]
    residuals = list(ar_actual - np.array(ar_fitted))
    ma_coeffs = fit_ma(residuals, q)

    predictions = []
    errors = list(residuals)  # continue the residual/error series forward

    for t in range(steps):
        diff_history = difference(history, d)
        ar_part = predict_ar(diff_history, ar_coeffs) if len(diff_history) >= p else 0.0
        ma_part = predict_ma(errors, ma_coeffs, q)
        yhat_diff = ar_part + ma_part

        forecast = inverse_difference(history, yhat_diff, d)
        predictions.append(forecast)
        history.append(forecast)

        # true error only computable while ground truth still exists
        idx = len(history) - 1
        if idx < len(series):
            error = series[idx] - forecast
        else:
            error = 0.0  # no ground truth beyond original series
        errors.append(error)

    return predictions


# ------------------------------------------------------------------
# 6. Run Model
# ------------------------------------------------------------------
p, d, q = 2, 1, 2
forecast_steps = 20

train = ts[:-forecast_steps]     # train on all but the last N points
test = ts[-forecast_steps:]      # holdout for evaluation

forecast = arima_forecast(train, p, d, q, steps=forecast_steps)

# ------------------------------------------------------------------
# 7. Evaluation (against a real, unseen holdout set)
# ------------------------------------------------------------------
mse = mean_squared_error(test, forecast)
mae = mean_absolute_error(test, forecast)
print("MSE:", mse)
print("MAE:", mae)

# ------------------------------------------------------------------
# 8. Visualization
# ------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(range(len(ts)), ts, label='Actual Data', color='blue')
plt.plot(range(len(train), len(train) + forecast_steps), forecast,
         label='Forecast', color='red')
plt.legend()
plt.title("ARIMA Forecast on Apple Stock Prices")
plt.xlabel("Time step")
plt.ylabel("Price (USD)")
plt.show()