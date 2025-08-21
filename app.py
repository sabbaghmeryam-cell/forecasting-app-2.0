import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="AI Forecasting Assistant", layout="wide")

st.title("🤖 AI Forecasting Assistant")
st.markdown('''
Upload your time series data (CSV or Excel), pick a model, and generate a forecast.
This app supports **ARIMA**, **Exponential Smoothing**, and a **Naive Seasonal** baseline.
''')

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Settings")
    horizon = st.number_input("Forecast horizon (periods)", min_value=1, value=30, step=1)
    test_size = st.number_input("Backtest size (last N periods)", min_value=12, value=60, step=1)
    model_name = st.selectbox("Model", ["ARIMA", "Exponential Smoothing", "Naive Seasonal"])
    seasonal_period = st.number_input("Seasonal period (e.g., 7=weekly, 12=monthly)", min_value=1, value=7, step=1)
    
    if model_name == "ARIMA":
        st.subheader("ARIMA Parameters")
        p = st.number_input("AR order (p)", min_value=0, value=1, step=1)
        d = st.number_input("Differencing (d)", min_value=0, value=1, step=1)
        q = st.number_input("MA order (q)", min_value=0, value=1, step=1)
        auto_arima = st.checkbox("Auto-select parameters", value=True)
    
    if model_name == "Exponential Smoothing":
        use_trend = st.checkbox("Use trend", value=True)
        use_seasonal = st.checkbox("Use seasonal", value=True)
    
    st.caption("Tip: If you're unsure, leave defaults as-is.")

uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

def try_parse_dates(series):
    try:
        return pd.to_datetime(series, errors="raise")
    except Exception:
        return None

def detect_columns(df):
    date_col = None
    value_col = None
    # Try common names first
    for c in df.columns:
        if c.lower() in ["date", "ds", "time", "timestamp"]:
            parsed = try_parse_dates(df[c])
            if parsed is not None:
                date_col = c
                break
    if date_col is None:
        # Fallback: first parsable date column
        for c in df.columns:
            parsed = try_parse_dates(df[c])
            if parsed is not None:
                date_col = c
                break
    # Numeric value column
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) > 0:
        # prefer a column named y/value/target
        preferred = [c for c in numeric_cols if c.lower() in ["y", "value", "target", "sales", "demand"]]
        value_col = preferred[0] if len(preferred) else numeric_cols[0]
    return date_col, value_col

def train_test_split_ts(df, test_size):
    return df.iloc[:-test_size], df.iloc[-test_size:]

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def check_stationarity(series):
    """Check if series is stationary using ADF test"""
    result = adfuller(series.dropna())
    return result[1] < 0.05  # p-value < 0.05 means stationary

def find_best_arima_params(series, max_p=3, max_d=2, max_q=3):
    """Simple grid search for ARIMA parameters"""
    best_aic = float('inf')
    best_params = (1, 1, 1)
    
    # Check if differencing is needed
    d_needed = 0
    temp_series = series.copy()
    while d_needed <= max_d and not check_stationarity(temp_series):
        d_needed += 1
        if d_needed <= max_d:
            temp_series = temp_series.diff().dropna()
    
    for p in range(max_p + 1):
        for d in range(min(d_needed + 1, max_d + 1)):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_params = (p, d, q)
                except:
                    continue
    
    return best_params

def fit_forecast_arima(train, horizon, p=1, d=1, q=1, auto_params=True):
    """Fit ARIMA model and generate forecast"""
    if auto_params:
        p, d, q = find_best_arima_params(train)
        st.write(f"Auto-selected ARIMA parameters: ({p}, {d}, {q})")
    
    model = ARIMA(train, order=(p, d, q))
    fitted = model.fit()
    
    # Generate forecast
    forecast = fitted.forecast(steps=horizon)
    conf_int = fitted.get_forecast(steps=horizon).conf_int()
    
    return fitted, forecast, conf_int

def fit_forecast_es(train, horizon, seasonal_period, use_trend=True, use_seasonal=True):
    """Fit Exponential Smoothing model and generate forecast"""
    trend = "add" if use_trend else None
    seasonal = "add" if use_seasonal and len(train) >= 2 * seasonal_period else None
    
    model = ExponentialSmoothing(
        train, 
        trend=trend, 
        seasonal=seasonal, 
        seasonal_periods=seasonal_period if seasonal else None,
        initialization_method="estimated"
    )
    fitted = model.fit(optimized=True)
    fc = fitted.forecast(horizon)
    return fitted, fc

def fit_forecast_naive_seasonal(train, horizon, seasonal_period):
    """Naive seasonal forecast - repeat last seasonal cycle"""
    if len(train) < seasonal_period:
        # If not enough data, use simple mean
        forecast_value = train.mean()
        forecast = pd.Series([forecast_value] * horizon)
    else:
        # Repeat last seasonal cycle
        last_season = np.array(train[-seasonal_period:])
        forecast_values = np.tile(last_season, int(np.ceil(horizon / seasonal_period)))[:horizon]
        forecast = pd.Series(forecast_values)
    
    return None, forecast

def backtest_forecast(series, model_name, seasonal_period, use_trend=True, use_seasonal=True, 
                     test_size=60, p=1, d=1, q=1, auto_params=True):
    """Perform backtesting to evaluate model performance"""
    if test_size <= 0 or test_size >= len(series):
        return None

    train, test = train_test_split_ts(series.to_frame(name="y"), test_size)
    train_y = train["y"]
    test_y = test["y"]

    try:
        if model_name == "ARIMA":
            _, fc, _ = fit_forecast_arima(train_y, len(test_y), p, d, q, auto_params)
            pred = pd.Series(fc, index=test.index)
        elif model_name == "Exponential Smoothing":
            _, fc = fit_forecast_es(train_y, len(test_y), seasonal_period, use_trend, use_seasonal)
            pred = pd.Series(fc.values if hasattr(fc, 'values') else fc, index=test.index)
        else:  # Naive Seasonal
            _, fc = fit_forecast_naive_seasonal(train_y, len(test_y), seasonal_period)
            pred = pd.Series(fc.values if hasattr(fc, 'values') else fc, index=test.index)

        metrics = {
            "MAPE (%)": mape(test_y, pred),
            "MAE": mean_absolute_error(test_y, pred),
            "RMSE": math.sqrt(mean_squared_error(test_y, pred)),
        }
        return metrics, test_y, pred
    except Exception as e:
        st.error(f"Error in backtesting: {str(e)}")
        return None

def plot_forecast(history, forecast, conf_int=None, title="Forecast"):
    """Plot historical data and forecast"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot history
    ax.plot(history.index, history.values, label="Historical Data", color='blue')
    
    # Plot forecast
    future_dates = pd.date_range(
        start=history.index[-1], 
        periods=len(forecast) + 1, 
        freq=pd.infer_freq(history.index) or 'D'
    )[1:]
    
    ax.plot(future_dates, forecast, label="Forecast", color='red', linestyle='--')
    
    # Plot confidence intervals if available
    if conf_int is not None:
        ax.fill_between(
            future_dates, 
            conf_int.iloc[:, 0], 
            conf_int.iloc[:, 1], 
            color='red', alpha=0.2, label='Confidence Interval'
        )
    
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def create_sample_data():
    """Create sample time series data"""
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Create a trend with seasonality and noise
    trend = np.linspace(100, 200, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)  # Yearly seasonality
    weekly = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 5, len(dates))
    
    values = trend + seasonal + weekly + noise
    
    return pd.DataFrame({'Date': dates, 'Value': values})

# Main app logic
if uploaded is None:
    st.info("Upload a CSV/Excel file or try the sample data below.")
    if st.button("Use sample data"):
        df = create_sample_data()
        st.success("Sample data loaded!")
    else:
        st.stop()
else:
    # Read uploaded file
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.success(f"File uploaded successfully! Shape: {df.shape}")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()

# Show data preview
st.subheader("Data Preview")
st.dataframe(df.head())

# Detect columns
date_col, value_col = detect_columns(df)

st.subheader("1) Select Columns")
c1, c2 = st.columns(2)
with c1:
    date_col = st.selectbox(
        "Date column", 
        options=list(df.columns), 
        index=list(df.columns).index(date_col) if date_col in df.columns else 0
    )
with c2:
    value_col = st.selectbox(
        "Value column", 
        options=list(df.columns), 
        index=list(df.columns).index(value_col) if value_col in df.columns else 0
    )

# Prepare time series
try:
    ser = df[[date_col, value_col]].copy()
    ser = ser.dropna()
    
    # Convert date column
    ser[date_col] = pd.to_datetime(ser[date_col], errors="coerce")
    ser = ser.dropna()
    ser = ser.sort_values(by=date_col)
    ser = ser.set_index(date_col)[value_col]
    
    # Handle duplicates by taking mean
    ser = ser.groupby(ser.index).mean()
    
    # Try to infer frequency
    freq = pd.infer_freq(ser.index)
    if freq is not None:
        ser = ser.asfreq(freq)
        # Forward fill missing values
        ser = ser.fillna(method='ffill').fillna(method='bfill')
    
    st.success(f"Time series prepared! Length: {len(ser)}, Frequency: {freq or 'Irregular'}")
    
except Exception as e:
    st.error(f"Error preparing time series: {str(e)}")
    st.stop()

# Show time series plot
st.subheader("Time Series Plot")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ser.index, ser.values)
ax.set_title("Time Series Data")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Backtesting
st.subheader("2) Model Evaluation (Backtest)")
if len(ser) > test_size:
    if model_name == "ARIMA":
        bt = backtest_forecast(ser, model_name, seasonal_period, test_size=test_size, 
                             p=p, d=d, q=q, auto_params=auto_arima)
    else:
        bt = backtest_forecast(ser, model_name, seasonal_period, use_trend, use_seasonal, test_size)
    
    if bt is not None:
        metrics, test_y, pred = bt
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAPE (%)", f"{metrics['MAPE (%)']:.2f}")
        with col2:
            st.metric("MAE", f"{metrics['MAE']:.2f}")
        with col3:
            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
        
        # Plot backtest results
        fig_bt, ax_bt = plt.subplots(figsize=(12, 6))
        ax_bt.plot(test_y.index, test_y.values, label="Actual", color='blue')
        ax_bt.plot(pred.index, pred.values, label="Predicted", color='red', linestyle='--')
        ax_bt.legend()
        ax_bt.set_title("Backtest: Actual vs Predicted")
        ax_bt.set_xlabel("Date")
        ax_bt.set_ylabel("Value")
        ax_bt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_bt)
    else:
        st.warning("Backtest failed. Check your parameters.")
else:
    st.warning(f"Not enough data for backtesting. Need at least {test_size + 1} data points.")

# Generate forecast
st.subheader("3) Generate Forecast")
try:
    if model_name == "ARIMA":
        model, forecast_values, conf_int = fit_forecast_arima(ser, horizon, p, d, q, auto_arima)
        plot_forecast(ser, forecast_values, conf_int, title=f"{model_name} Forecast ({horizon} steps)")
    elif model_name == "Exponential Smoothing":
        model, forecast_values = fit_forecast_es(ser, horizon, seasonal_period, use_trend, use_seasonal)
        plot_forecast(ser, forecast_values, title=f"{model_name} Forecast ({horizon} steps)")
    else:  # Naive Seasonal
        model, forecast_values = fit_forecast_naive_seasonal(ser, horizon, seasonal_period)
        plot_forecast(ser, forecast_values, title=f"{model_name} Forecast ({horizon} steps)")
    
    st.success("Forecast generated successfully!")
    
except Exception as e:
    st.error(f"Error generating forecast: {str(e)}")
    st.stop()

# Download forecast
st.subheader("4) Download Forecast")
try:
    # Create future dates
    last_date = ser.index[-1]
    if freq is not None:
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
    else:
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq='D')[1:]
    
    # Create output dataframe
    out_df = pd.DataFrame({
        "date": future_dates,
        "forecast": forecast_values.values if hasattr(forecast_values, 'values') else forecast_values
    })
    
    # Show forecast table
    st.dataframe(out_df)
    
    # Download button
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Forecast CSV", 
        data=csv_bytes, 
        file_name="forecast.csv", 
        mime="text/csv"
    )
    
except Exception as e:
    st.error(f"Error preparing download: {str(e)}")

st.divider()
st.markdown("Built with ❤️ using Streamlit and statsmodels.")
st.caption("Note: This version uses statsmodels ARIMA instead of pmdarima to avoid dependency issues.")