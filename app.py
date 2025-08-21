import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# Page config
st.set_page_config(page_title="AI Forecasting Assistant", layout="wide")

st.title("🤖 AI Forecasting Assistant")
st.markdown("Upload your time series data and generate forecasts using simple statistical methods.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    forecast_periods = st.number_input("Forecast periods", min_value=1, max_value=365, value=30)
    method = st.selectbox("Forecasting method", ["Simple Average", "Linear Trend", "Seasonal Naive"])
    seasonal_period = st.number_input("Seasonal period", min_value=2, value=7)

def create_sample_data():
    """Create sample time series data"""
    dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
    trend = np.linspace(100, 150, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    noise = np.random.normal(0, 3, len(dates))
    values = trend + seasonal + noise
    return pd.DataFrame({'Date': dates, 'Value': values})

def simple_forecast(data, periods, method, seasonal_period):
    """Simple forecasting methods"""
    values = data['Value'].values
    
    if method == "Simple Average":
        # Use last 30 days average
        forecast_value = np.mean(values[-30:])
        forecast = [forecast_value] * periods
        
    elif method == "Linear Trend":
        # Fit linear trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        
        # Extend trend
        future_x = np.arange(len(values), len(values) + periods)
        forecast = np.polyval(coeffs, future_x)
        
    else:  # Seasonal Naive
        # Repeat last seasonal pattern
        if len(values) >= seasonal_period:
            last_season = values[-seasonal_period:]
            forecast = np.tile(last_season, int(np.ceil(periods / seasonal_period)))[:periods]
        else:
            forecast = [np.mean(values)] * periods
    
    return forecast

def plot_forecast(historical, forecast, title="Forecast"):
    """Plot historical data and forecast"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    ax.plot(historical['Date'], historical['Value'], label='Historical', color='blue', linewidth=2)
    
    # Create future dates
    last_date = historical['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast), freq='D')
    
    # Plot forecast
    ax.plot(future_dates, forecast, label='Forecast', color='red', linestyle='--', linewidth=2)
    
    # Styling
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Column selection
        st.subheader("Select Columns")
        col1, col2 = st.columns(2)
        
        with col1:
            date_col = st.selectbox("Date column", df.columns)
        with col2:
            value_col = st.selectbox("Value column", df.columns)
        
        # Process data
        if st.button("Generate Forecast"):
            try:
                # Prepare data
                forecast_df = df[[date_col, value_col]].copy()
                forecast_df = forecast_df.dropna()
                forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])
                forecast_df = forecast_df.sort_values(date_col)
                forecast_df.columns = ['Date', 'Value']
                
                # Generate forecast
                forecast = simple_forecast(forecast_df, forecast_periods, method, seasonal_period)
                
                # Plot results
                st.subheader("Forecast Results")
                fig = plot_forecast(forecast_df, forecast, f"{method} Forecast")
                st.pyplot(fig)
                
                # Create forecast dataframe
                last_date = forecast_df['Date'].iloc[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast), freq='D')
                forecast_df_output = pd.DataFrame({
                    'Date': future_dates,
                    'Forecast': forecast
                })
                
                st.subheader("Forecast Data")
                st.dataframe(forecast_df_output)
                
                # Download button
                csv = forecast_df_output.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name="forecast.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

else:
    st.info("Please upload a CSV file to get started, or try the sample data below.")
    
    if st.button("Use Sample Data"):
        # Generate and use sample data
        sample_df = create_sample_data()
        
        st.subheader("Sample Data Preview")
        st.dataframe(sample_df.head())
        
        # Show sample data plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sample_df['Date'], sample_df['Value'], linewidth=2)
        ax.set_title('Sample Time Series Data', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Generate forecast with sample data
        if st.button("Generate Sample Forecast"):
            forecast = simple_forecast(sample_df, forecast_periods, method, seasonal_period)
            
            st.subheader("Sample Forecast")
            fig = plot_forecast(sample_df, forecast, f"{method} Sample Forecast")
            st.pyplot(fig)
            
            # Show some metrics
            st.subheader("Forecast Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Forecast Periods", len(forecast))
            with col2:
                st.metric("Avg Forecast Value", f"{np.mean(forecast):.2f}")
            with col3:
                st.metric("Method Used", method)

# Footer
st.divider()
st.markdown("Built with ❤️ using Streamlit")
st.caption("This is a simplified forecasting app using basic statistical methods.")