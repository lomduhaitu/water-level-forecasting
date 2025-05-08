import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from io import StringIO

# App config
st.set_page_config(layout="wide", page_title="Water Level Forecaster")
st.title("🌊 Hybrid Water Level Forecasting")

@st.cache_resource
def load_models():
    """Load pretrained models"""
    return (
        SARIMAXResults.load("assets/my_sarima_model.pkl"),
        load_model("assets/my_lstm_model.h5", compile=False),
        joblib.load("assets/scaler.pkl")
    )

# Sidebar controls
st.sidebar.header("Configuration")
forecast_days = st.sidebar.slider("Forecast days", 7, 365, 30)
input_method = st.sidebar.radio("Input method", ["CSV Upload", "Manual Entry"])

# Main content
if input_method == "CSV Upload":
    st.header("1. Upload Historical Data")
    uploaded_file = st.file_uploader("CSV with Date and Water_Level_m columns", type="csv")
    
    if uploaded_file:
        user_data = pd.read_csv(uploaded_file, parse_dates=["Date"])
        user_data.set_index("Date", inplace=True)
else:
    st.header("1. Enter Historical Data")
    num_days = st.number_input("Days of history (min 30)", min_value=30, value=30)
    
    dates = []
    levels = []
    cols = st.columns(2)
    for i in range(num_days):
        with cols[0]:
            dates.append(st.date_input(f"Day {i+1}", key=f"d{i}"))
        with cols[1]:
            levels.append(st.number_input(f"Level (m) {i+1}", key=f"l{i}"))
    
    if st.button("Create Dataset"):
        user_data = pd.DataFrame({
            "Date": dates,
            "Water_Level_m": levels
        }).set_index("Date")

# Forecasting
if 'user_data' in locals() and len(user_data) >= 30:
    st.header("2. Forecast Results")
    
    # Preprocess
    user_data = user_data.asfreq('d').ffill()
    
    # Load models
    sarima, lstm, scaler = load_models()
    
    # Generate forecast (hybrid approach)
    with st.spinner("Computing forecast..."):
        # SARIMA prediction
        sarima_pred = sarima.get_forecast(steps=forecast_days).predicted_mean
        
        # LSTM residual prediction
        last_30 = user_data["Water_Level_m"].values[-30:].reshape(-1, 1)
        last_30_scaled = scaler.transform(last_30)
        lstm_residual = lstm.predict(last_30_scaled[np.newaxis, ...]).flatten()
        
        # Combine
        forecast = sarima_pred + lstm_residual[:forecast_days]
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    user_data["Water_Level_m"].plot(ax=ax, label="Historical")
    forecast.plot(ax=ax, label="Forecast", color='red')
    ax.set_title(f"{forecast_days}-Day Water Level Forecast")
    ax.legend()
    st.pyplot(fig)
    
    # Export
    csv = forecast.reset_index().to_csv(index=False)
    st.download_button(
        "Download Forecast",
        csv,
        "water_level_forecast.csv",
        "text/csv"
    )

elif 'user_data' in locals():
    st.error("❌ Need at least 30 days of data")
else:
    st.info("ℹ️ Upload data or enter readings to begin")
