# ARIMA from Scratch (Python + NumPy)

# Objective
The goal of this project was to implement the ARIMA (AutoRegressive Integrated Moving Average) model **from scratch** in Python using only `numpy` and `pandas` (without high-level libraries like `statsmodels`).  

This demonstrates understanding of **differencing, autoregression, and moving average** in time series forecasting.

---

# Dataset
- Dataset: **Apple Stock Prices**  
- Source: [Apple Finance Dataset](https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv)  
- Column Used: `AAPL.Close` (daily closing price)  

---

# Implementation
- Differencing for stationarity (**I part**)  
- Autoregression (AR) using lag values  
- Moving Average (MA) using residuals  
- Combined ARIMA(p,d,q) model for forecasting  
- Evaluation using **MSE** and **MAE**  

---

# Results
- **First 10 values:** `[27.950001 28.122499 27.8575 28.385   28.5175 28.5625 28.3675 28.23 28.34 28.122499]`  
- **MSE:** (printed when running code)  
- **MAE:** (printed when running code)  

Forecast plot shows:  
- Blue line = Actual stock closing prices  
- Red line = Predicted future values  

---

# Tools Used
- Python 3.13  
- NumPy  
- Pandas  
- Matplotlib  
- scikit-learn (for evaluation metrics only)  

---

# How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/harshithadevarinti/ARIMA_from_scratch.git
   cd ARIMA_from_scratch
