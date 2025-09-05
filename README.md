# ARIMA from Scratch (Python + NumPy)

## 📌 Objective
The goal of this project was to implement the ARIMA (AutoRegressive Integrated Moving Average) model **from scratch** in Python using only `numpy` and `pandas` (without high-level libraries like `statsmodels`).

## 📂 Dataset
- Dataset: Airline Passengers (monthly total international airline passengers, 1949–1960)
- Source: [Airline Passengers Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv)

## ⚙️ Implementation
- Differencing for stationarity (I part)  
- Autoregression (AR) using lag values  
- Moving Average (MA) using residuals  
- Combined ARIMA(p,d,q) forecasting  
- Evaluation using **MSE** and **MAE**

## 📊 Results
- **First 10 values:** `[112. 118. 132. 129. 121. 135. 148. 148. 136. 119.]`  
- **MSE:** 6069.79  
- **MAE:** 60.07  

The forecast plot shows predicted values vs actual values.

## 🛠️ Tools Used
- Python 3.13
- NumPy
- Pandas
- Matplotlib
- scikit-learn (for evaluation metrics only)

## 🚀 How to Run
```bash
git clone https://github.com/your-username/ARIMA-from-scratch.git
cd ARIMA-from-scratch
python arima_from_scratch.py
