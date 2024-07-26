import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import json
from tensorflow.keras.models import model_from_json, Sequential

def load_model_without_time_major(filepath):
    with h5py.File(filepath, 'r') as f:
        model_config = f.attrs.get('model_config')
        if (model_config is None):
            raise ValueError("No model config found in the file.")
        
        model_config = json.loads(model_config)
        
        # Recursively remove 'time_major' from config
        def remove_time_major(config):
            if isinstance(config, dict):
                if 'time_major' in config:
                    del config['time_major']
                for key, value in config.items():
                    remove_time_major(value)
            elif isinstance(config, list):
                for item in config:
                    remove_time_major(item)
        
        remove_time_major(model_config)
        
        model_json = json.dumps(model_config)
        
        # Specify the custom object during deserialization
        custom_objects = {'Sequential': Sequential}
        model = model_from_json(model_json, custom_objects=custom_objects)
        
        # Load weights into the model
        model.load_weights(filepath)
    
    return model

def load_model_and_scalers():
    # Load the saved model and scalers
    model = load_model_without_time_major('model_final.h5')
    
    with open('feature_scaler.pkl', 'rb') as f:
        feature_scaler = pickle.load(f)
    
    with open('target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    
    return model, feature_scaler, target_scaler

def create_fourier_series(df, period, order):
    t = np.arange(len(df))
    for k in range(1, order + 1):
        df[f'sin_{period}_{k}'] = np.sin(2 * np.pi * k * t / period)
        df[f'cos_{period}_{k}'] = np.cos(2 * np.pi * k * t / period)
    return df

def preprocess_data(df):
    # Calculate IQR
    Q1 = df['avg_cpu'].quantile(0.25)
    Q3 = df['avg_cpu'].quantile(0.75)
    IQR = Q3 - Q1

    # Define a threshold to identify anomalies (e.g., 1.5 * IQR)
    iqr_threshold = 1.5
    lower_bound = Q1 - iqr_threshold * IQR
    upper_bound = Q3 + iqr_threshold * IQR
    anomalies_iqr = df[(df['avg_cpu'] < lower_bound) | (df['avg_cpu'] > upper_bound)]

    # Mark anomalies
    df['anomaly'] = (df['avg_cpu'] < lower_bound) | (df['avg_cpu'] > upper_bound)

    # Interpolating anomalies
    df['avg_cpu'] = np.where(df['anomaly'] == True, np.nan, df['avg_cpu'])
    df['avg_cpu'] = df['avg_cpu'].interpolate(method='time')

    # Drop the anomaly column as it's no longer needed
    df = df.drop(columns=['anomaly'])

    # Add polynomial trends
    time_numeric = np.arange(len(df)).reshape(-1, 1)
    degree = 2
    poly = PolynomialFeatures(degree=degree)
    time_poly = poly.fit_transform(time_numeric)
    model = LinearRegression()
    model.fit(time_poly, df['avg_cpu'])
    trend = model.predict(time_poly)
    df['trend'] = trend

    # Add Fourier series
    df = create_fourier_series(df, period=24, order=3)
    df = create_fourier_series(df, period=24 * 7, order=3)
    df['trend'] = trend

    # Add lag features
    max_lag = 36
    for lag in range(1, max_lag + 1):
        df[f'avg_cpu_lag_{lag}'] = df['avg_cpu'].shift(lag)
    df.dropna(inplace=True)
    
    return df

def create_sliding_windows(data, window_size):
    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i+window_size])
    return np.array(X)

def main():
    # Load model and scalers
    model, feature_scaler, target_scaler = load_model_and_scalers()

    # Main Streamlit app
    st.title('CPU Usage Prediction App')

    # File uploader for the input data
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the uploaded CSV file
        test_input_df = pd.read_csv(uploaded_file, parse_dates=["time"], index_col="time")

        # Display the input data
        st.subheader('Input Data')
        st.write(test_input_df.head())

        # Preprocess the input data
        test_input_df = preprocess_data(test_input_df)

        # Normalize the features using the saved scaler
        features = test_input_df.drop(columns=['avg_cpu'])
        features_scaled = feature_scaler.transform(features)

        # Create sliding windows
        window_size = 50
        X_input = create_sliding_windows(features_scaled, window_size)

        # Check if we have enough data for sliding windows
        if len(X_input) == 0:
            st.error("Not enough data to create sliding windows. Ensure you have at least 'window_size' rows after preprocessing.")
        else:
            # Make predictions
            y_pred = model.predict(X_input[-1].reshape(1, window_size, -1))
            y_pred = target_scaler.inverse_transform(y_pred)

            # Generate the timestamp for the predicted value
            last_actual_timestamp = test_input_df.index[-1]
            predicted_timestamp = last_actual_timestamp + pd.Timedelta(minutes=5)

            # Create the predicted_df DataFrame
            predicted_df = pd.DataFrame({
                'time': [predicted_timestamp],
                'predicted_avg_cpu': y_pred.flatten()
            })
            predicted_df.set_index('time', inplace=True)

            # Generate the advice based on the prediction
            predicted_value = predicted_df['predicted_avg_cpu'].iloc[0]
            advice = f"<p style='font-size:24px;'>The predicted CPU usage at {predicted_timestamp} is {predicted_value:.2f}. " \
                     "Administrators should monitor the system closely during this period.</p>"

            # Display the predictions
            st.subheader('Predicted CPU Usage for the Next Time Point')
            st.write(predicted_df)

            # Plot the predictions
            st.subheader('Prediction Plot')
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.scatter(predicted_df.index, predicted_df['predicted_avg_cpu'], label='Predicted CPU Usage', color='orange')
            ax.set_title('Predicted CPU Usage for the Next Time Point', fontsize=24)
            ax.set_xlabel('Time', fontsize=20)
            ax.set_ylabel('Predicted Average CPU Usage', fontsize=20)
            ax.legend(fontsize=20)
            ax.grid(True)
            st.pyplot(fig)

            # Display the advice
            st.subheader('Advice')
            st.markdown(advice, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
