import os
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from src.models.utils import load_model

API_URL = os.getenv('API_URL', 'http://api:8000')


st.title("Credit Risk Forecast Dashboard")
model = st.selectbox("Model", ["ARIMA", "XGB"])

horizon = st.slider("Forecast Horizon (months)", 1, 24, 6)

if st.button("Get Forecast"):
    with st.spinner("Fetching forecast..."):
           response = requests.post(
        f"{API_URL}/forecast",
        json={"horizon": horizon, "model": model}
    )
    if response.ok:
        data = response.json()
        dates = pd.date_range(end=datetime.today(), periods=horizon+1, freq='M')[1:]
   
       
        # derive how many points we actually got back
        if model == "ARIMA":
          n = len(data['mean'])
        else:
          n = len(data['xgb'])

        # build exactly n monthly end-of-month dates
        dates = pd.date_range(end=datetime.today(), periods=n+1, freq='M')[1:]

        if model == "ARIMA":
         df = pd.DataFrame({
            'date':   dates,
            'mean':   data['mean'],
            'lower':  data['lower'],
            'upper':  data['upper']
         }).set_index('date')
         st.line_chart(df['mean'])
         st.area_chart(df[['lower','upper']])
        else:  # XGB
         df = pd.DataFrame({'date': dates, 'xgb': data['xgb']}).set_index('date')
         st.line_chart(df['xgb'])
        if data.get("mape") is not None:
         st.markdown(f"**In-sample MAPE for {model}:** {data['mape']:.2%}")
    else:
            st.error(f"Error: {response.text}")

st.markdown("---")
st.write("Upload your dataset to see custom forecast")
uploaded_file = st.file_uploader("Choose CSV", type="csv")
if uploaded_file:
    df_input = pd.read_csv(uploaded_file, parse_dates=['timestamp'], index_col='timestamp')
    if st.button("Forecast on uploaded data"):
        model = load_model('arima')
        pred = model.get_forecast(steps=horizon)
        forecast = pred.predicted_mean
        dates = pd.date_range(end=df_input.index[-1], periods=horizon+1, freq='M')[1:]
        df_res = pd.DataFrame({'date': dates, 'forecast': forecast}).set_index('date')
        st.line_chart(df_res)