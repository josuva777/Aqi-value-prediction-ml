import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import plotly.express as px
# from streamlit_lottie import st_lottie # Commented out as animations caused issues
import datetime

# -------------------
# Load models and data with caching
# -------------------
@st.cache_resource
def load_resources():
    reg_model, clf_model, preprocessor, label_encoder = None, None, None, None
    models_loaded = False
    try:
        reg_model = joblib.load('regression_model.joblib')
        clf_model = joblib.load('classification_model.joblib')
        preprocessor = joblib.load('preprocessor.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        models_loaded = True
    except FileNotFoundError:
        st.error("Error loading model files. Please ensure model files are in the directory.")

    df_app = pd.DataFrame()
    data_loaded = False
    try:
        df_app = pd.read_csv('model.csv', parse_dates=['date'])
        data_loaded = True
    except FileNotFoundError:
        st.error("Error loading data file. Please ensure 'model.csv' is in the directory.")

    return reg_model, clf_model, preprocessor, label_encoder, df_app, models_loaded, data_loaded

reg_model, clf_model, preprocessor, label_encoder, df_app, models_loaded, data_loaded = load_resources()

# -------------------
# Prediction function (same as your Streamlit version)
# -------------------
def predict_aqi_and_status(city, future_date, historical_data, reg_model, clf_model, preprocessor, label_encoder,
                           temp_input=None, humidity_input=None, wind_speed_input=None, precipitation_input=None):
    future_date = pd.to_datetime(future_date)

    if city not in historical_data['city'].unique():
        return None, f"Error: City '{city}' not found in historical data."

    city_historical_data = historical_data[historical_data['city'] == city].sort_values(by='date').copy()

    days_needed_for_features = 3
    start_date_for_features = future_date - pd.Timedelta(days=days_needed_for_features)

    recent_city_data = city_historical_data[city_historical_data['date'] >= start_date_for_features].copy()

    future_data = pd.DataFrame({'date': [future_date], 'city': [city]})

    future_data['temperature'] = temp_input if temp_input is not None else np.nan
    future_data['humidity'] = humidity_input if humidity_input is not None else np.nan
    future_data['wind_speed'] = wind_speed_input if wind_speed_input is not None else np.nan
    future_data['precipitation'] = precipitation_input if precipitation_input is not None else np.nan

    cols_for_concat = ['date', 'city', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
                       'Benzene', 'Toluene', 'Xylene', 'aqi_value',
                       'temperature', 'humidity', 'wind_speed', 'precipitation']

    for col in cols_for_concat:
        if col not in recent_city_data.columns:
            recent_city_data[col] = np.nan

    recent_city_data_subset = recent_city_data[cols_for_concat].copy()


    combined_data = pd.concat([recent_city_data_subset, future_data], ignore_index=True)

    combined_data['year'] = combined_data['date'].dt.year
    combined_data['month'] = combined_data['date'].dt.month
    combined_data['dayofweek'] = combined_data['date'].dt.dayofweek
    combined_data['dayofyear'] = combined_data['date'].dt.dayofyear
    combined_data['weekofyear'] = combined_data['date'].dt.isocalendar().week.astype(int)

    combined_data = combined_data.sort_values(by='date').reset_index(drop=True)

    features_to_engineer = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
                           'Benzene', 'Toluene', 'Xylene', 'aqi_value',
                           'temperature', 'humidity', 'wind_speed', 'precipitation']

    for feature in features_to_engineer:
        combined_data[f'{feature}_lag1'] = combined_data.groupby('city')[feature].shift(1)
        combined_data[f'{feature}_rolling_mean_3d'] = combined_data.groupby('city')[feature].rolling(window=3, min_periods=1).mean().reset_index(drop=True)
        combined_data[f'{feature}_rolling_std_3d'] = combined_data.groupby('city')[feature].rolling(window=3, min_periods=1).std().reset_index(drop=True)

    combined_data['PM2.5_temp_interaction'] = combined_data['PM2.5'] * combined_data['temperature']
    combined_data['PM10_humidity_interaction'] = combined_data['PM10'] * combined_data['humidity']
    combined_data['O3_wind_interaction'] = combined_data['O3'] * combined_data['wind_speed']


    future_data_row = combined_data[combined_data['date'] == future_date].copy()

    num_features_trained = [col for col in historical_data.columns if col.endswith(('_lag1', '_rolling_mean_3d', '_rolling_std_3d'))]
    num_features_trained.extend(['PM2.5_temp_interaction', 'PM10_humidity_interaction', 'O3_wind_interaction'])
    num_features_trained.extend(['year', 'month', 'dayofweek', 'dayofyear', 'weekofyear'])
    cat_features_trained = ['city']
    all_features_trained = num_features_trained + cat_features_trained


    for col in all_features_trained:
        if col not in future_data_row.columns:
            future_data_row[col] = np.nan

    future_data_processed = future_data_row[all_features_trained]


    numerical_features_only = [f for f in all_features_trained if f != 'city']
    if not historical_data.empty:
        historical_means = historical_data[numerical_features_only].mean()
        future_data_processed[numerical_features_only] = future_data_processed[numerical_features_only].fillna(historical_means)
    else:
        future_data_processed[numerical_features_only] = future_data_processed[numerical_features_only].fillna(0)


    try:
        future_data_transformed = preprocessor.transform(future_data_processed)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None, "Error during preprocessing"


    if reg_model and clf_model and label_encoder:
        predicted_aqi = reg_model.predict(future_data_transformed)[0]
        predicted_status_encoded = clf_model.predict(future_data_transformed)[0]
        predicted_status = label_encoder.inverse_transform([predicted_status_encoded])[0]
        return predicted_aqi, predicted_status
    else:
        return None, "Models not loaded. Cannot make prediction."

# -------------------
# Weather API (Open-Meteo) for all cities
# -------------------
coords = {
    "Delhi": (28.61, 77.23),
    "Mumbai": (19.07, 72.87),
    "Chennai": (13.08, 80.27),
    "Bengaluru": (12.97, 77.59),
    "Kolkata": (22.57, 88.36),
    "Ahmedabad": (23.02, 72.57),
    "Hyderabad": (17.38, 78.48),
    "Pune": (18.52, 73.85),
    "Surat": (21.17, 72.83),
    "Jaipur": (26.91, 75.78),
    "Kanpur": (26.44, 80.36),
    "Lucknow": (26.84, 80.94),
    "Nagpur": (21.14, 79.08),
    "Visakhapatnam": (17.68, 83.21),
    "Bhopal": (23.25, 77.41),
    "Patna": (25.59, 85.13),
    "Vadodara": (22.30, 73.18),
    "Ghaziabad": (28.66, 77.42),
    "Ludhiana": (30.90, 75.85),
    "Agra": (27.17, 78.00),
    "Nashik": (19.99, 73.78),
    "Faridabad": (28.40, 77.31),
    "Meerut": (28.98, 77.70),
    "Rajkot": (22.29, 70.91),
    "Kalyan-Dombivali": (19.21, 73.09),
    "Vasai-Virar": (19.37, 72.81),
    "Varanasi": (25.31, 82.97),
    "Srinagar": (34.08, 74.80),
    "Aurangabad": (19.87, 75.34),
    "Dhanbad": (23.79, 86.43),
    "Amritsar": (31.63, 74.87),
    "Thane": (19.21, 72.97),
    "Howrah": (22.59, 88.26),
    "Allahabad": (25.43, 81.84),
    "Jabalpur": (23.16, 79.93),
    "Gwalior": (26.21, 78.17),
    "Vijayawada": (16.50, 80.64),
    "Jodhpur": (26.23, 73.02),
    "Raipur": (21.25, 81.62),
    "Kota": (25.21, 75.86)
}

@st.cache_data(ttl=300)
def get_weather(city):
    lat, lon = coords.get(city, (28.61, 77.23))  # Default Delhi
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.warning(f"Weather API error for {city}: {e}")
    return None

# -------------------
# Streamlit UI
# -------------------

st.set_page_config(page_title="AQI Prediction App", page_icon="🤖", layout="wide")

# Removed the problematic st.markdown block with custom CSS

st.title("🏙️ Urban Air Quality Dashboard")
st.markdown("### Predicting AQI and Visualizing Trends")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data Explorer", "☀️ Current Weather"])

# --- Prediction Tab ---
with tab1:
    st.header("Predict Air Quality")
    if not (models_loaded and data_loaded):
        st.warning("Models or data not loaded. AQI prediction is disabled.")
    else:
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.subheader("Prediction Inputs")
            cities = df_app['city'].unique()
            cities.sort()
            pred_city = st.selectbox("Select City:", cities, key='pred_city')
            min_date = df_app['date'].min().date()
            pred_date = st.date_input("Select Date:", datetime.date.today(), min_value=min_date, key='pred_date')

            st.markdown("---")
            st.subheader("Weather Conditions (Optional Input):")
            st.info("Provide estimated weather conditions for the prediction date. If left blank, historical averages will be used.")

            temp_input = st.number_input("Temperature (°C):", value=None, format="%.2f", key='pred_temp')
            humidity_input = st.number_input("Humidity (%):", value=None, format="%.2f", key='pred_humidity')
            wind_speed_input = st.number_input("Wind Speed (km/h):", value=None, format="%.2f", key='pred_wind')
            precipitation_input_display = st.selectbox("Precipitation:", ["No", "Yes"], index=0, key='pred_precip')
            precipitation_input = 1 if precipitation_input_display == "Yes" else 0

            if st.button("✨ Get Prediction"):
                with st.spinner("Predicting..."):
                    predicted_aqi, predicted_status = predict_aqi_and_status(
                        pred_city,
                        pred_date,
                        df_app,
                        reg_model,
                        clf_model,
                        preprocessor,
                        label_encoder,
                        temp_input=temp_input if temp_input is not None else None,
                        humidity_input=humidity_input if humidity_input is not None else None,
                        wind_speed_input=wind_speed_input if wind_speed_input is not None else None,
                        precipitation_input=precipitation_input
                    )
                if predicted_aqi is not None and isinstance(predicted_status, str):
                    st.success(f"Predicted AQI Value for {pred_city} on {pred_date}: **{predicted_aqi:.2f}**")
                    st.info(f"Predicted AQI Status for {pred_city} on {pred_date}: **{predicted_status}**")
                else:
                    st.error(f"Prediction Error: {predicted_status}")

        with col2:
            st.subheader("Prediction Explanation")
            st.markdown("""
            This prediction uses historical air quality data and optional weather inputs to forecast the AQI value and status.
            You can provide weather conditions for the prediction date or leave them blank to use historical averages.
            """)

# --- Data Explorer Tab ---
with tab2:
    st.header("Data Explorer")
    if not data_loaded:
        st.warning("Historical data not loaded. Data visualization is disabled.")
    else:
        cities = df_app['city'].unique()
        cities.sort()
        viz_city = st.selectbox("Select City:", cities, key='viz_city')
        # Filter columns for visualization (exclude engineered features)
        viz_columns = [col for col in df_app.columns if df_app[col].dtype in ['float64', 'int64']
                       and not any(suffix in col for suffix in ['_lag1', '_rolling_mean_3d', '_rolling_std_3d', '_interaction', 'year', 'month', 'dayofweek', 'dayofyear', 'weekofyear', 'city_encoded'])]
        viz_columns.sort()
        viz_column = st.selectbox("Select Metric:", viz_columns, key='viz_column')
        chart_type = st.selectbox("Select Chart Type:", ["Line Plot", "Histogram", "Box Plot"], key='chart_type')

        # Add a button to generate the graph
        if st.button("Generate Graph"):
            filtered_df = df_app[df_app['city'] == viz_city].copy()
            if chart_type == "Line Plot":
                filtered_df = filtered_df.sort_values(by='date')
                fig = px.line(filtered_df, x='date', y=viz_column, title=f'{viz_column} Trend for {viz_city}')
            elif chart_type == "Histogram":
                fig = px.histogram(filtered_df, x=viz_column, nbins=50, title=f'Distribution of {viz_column} in {viz_city}')
            else:  # Box Plot
                fig = px.box(filtered_df, y=viz_column, title=f'Distribution of {viz_column} in {viz_city}')
            fig.update_layout(template='plotly_dark', xaxis_title='Date' if chart_type == "Line Plot" else viz_column, yaxis_title=viz_column)
            st.plotly_chart(fig, use_container_width=True)

# --- Current Weather Tab ---
with tab3:
    st.header("☀️ Current Weather Check")
    if not data_loaded:
        st.warning("Historical data not loaded. City list for weather check is unavailable.")
    else:
        cities = df_app['city'].unique()
        cities.sort()
        weather_city = st.selectbox("Select City for Weather:", cities, key='weather_city')

        if st.button("Refresh Weather"):
            st.rerun()

        weather_data = get_weather(weather_city)
        if weather_data and "current_weather" in weather_data:
            weather = weather_data["current_weather"]
            col1, col2, col3 = st.columns(3)
            col1.metric("🌡️ Temperature (°C)", weather.get("temperature", "N/A"))
            col2.metric("💨 Wind Speed (km/h)", weather.get("windspeed", "N/A"))
            time_str = weather.get("time", "N/A")
            try:
                time_obj = pd.to_datetime(time_str)
                time_str = time_obj.strftime('%Y-%m-%d %H:%M')
            except Exception:
                pass
            col3.metric("🕒 Time", time_str)
        else:
            st.info("Could not retrieve current weather data for this city.")

# Note: The code below this point seems incomplete or remnants of previous attempts.
# The primary Streamlit app logic is above.
# If you intended to add more here, please provide the complete code.
