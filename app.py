import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Urban Air Quality Forecasting",
    layout="wide"
)

st.title("🌍 Urban Air Quality Forecasting System")
st.write("Predict Air Quality Index (AQI) and visualize pollution trends")

# Load trained model
model = joblib.load("aqi_prediction_model.pkl")

# AQI category function
def aqi_category(aqi):
    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Satisfactory 🟡"
    elif aqi <= 200:
        return "Moderate 🟠"
    elif aqi <= 300:
        return "Poor 🔴"
    elif aqi <= 400:
        return "Very Poor 🟣"
    else:
        return "Severe ⚫"

# File upload
uploaded_file = st.file_uploader("Upload Air Quality CSV File", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file, sep=';')

    # Drop empty columns
    df.dropna(axis=1, how='all', inplace=True)

    # Combine Date and Time
    df['Datetime'] = pd.to_datetime(
        df['Date'] + " " + df['Time'],
        format="%d/%m/%Y %H.%M.%S",
        errors='coerce'
    )

    df.drop(['Date', 'Time'], axis=1, inplace=True)

    # Convert numeric columns
    numeric_cols = ['CO(GT)', 'NO2(GT)', 'T', 'RH', 'AH']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.replace(-200, pd.NA, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Feature engineering
    df['Hour'] = df['Datetime'].dt.hour
    df['Day'] = df['Datetime'].dt.day
    df['Month'] = df['Datetime'].dt.month

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Select latest row for prediction
    input_features = df[['CO(GT)', 'NO2(GT)', 'T', 'RH', 'AH', 'Hour', 'Day', 'Month']].iloc[-1:]
    predicted_aqi = model.predict(input_features)[0]

    st.subheader("📈 AQI Prediction")
    st.metric("Predicted AQI", round(predicted_aqi, 2))

    category = aqi_category(predicted_aqi)
    st.success(f"AQI Category: {category}")

    if predicted_aqi > 200:
        st.error("⚠️ Health Alert: Avoid outdoor activities!")

    # Trend visualization
    st.subheader("📉 CO Pollution Trend")
    fig, ax = plt.subplots()
    ax.plot(df['Datetime'], df['CO(GT)'])
    ax.set_xlabel("Datetime")
    ax.set_ylabel("CO(GT)")
    st.pyplot(fig)
