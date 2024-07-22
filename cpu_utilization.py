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
from tensorflow.keras.models import model_from_json

def load_model_without_time_major(filepath):
    with h5py.File(filepath, 'r') as f:
        model_config = f.attrs.get('model_config')
        if model_config is None:
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
        model = model_from_json(model_json)
        
        # Load weights into the model
        model.load_weights(filepath)
    
    return model

def load_model_and_scalers():
    # Load the saved model and scalers
    model = load_model_without_time_major('best_model.h5')
    
    with open('feature_scaler.pkl', 'rb') as f:
        feature_scaler = pickle.load(f)
    
    with open('target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    
    return model, feature_scaler, target_scaler

def preprocess_data(df):
    # Placeholder function for data preprocessing
    return df

def create_sliding_windows(data, window_size):
    # Placeholder function for creating sliding windows
    return np.array([data[i:i+window_size] for i in range(len(data) - window_size + 1)])

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

            # Generate the timestamps for the predicted values
            last_actual_timestamp = test_input_df.index[-1]
            predicted_timestamps = pd.date_range(start=last_actual_timestamp, periods=13, freq='5T')[1:]

            # Create the predicted_df DataFrame
            predicted_df = pd.DataFrame({
                'time': predicted_timestamps,
                'predicted_avg_cpu': y_pred.flatten()
            })
            predicted_df.set_index('time', inplace=True)

            # Identify the highest and lowest predicted values and their corresponding times
            max_pred = predicted_df['predicted_avg_cpu'].max()
            max_pred_time = predicted_df['predicted_avg_cpu'].idxmax()
            min_pred = predicted_df['predicted_avg_cpu'].min()
            min_pred_time = predicted_df['predicted_avg_cpu'].idxmin()

            # Generate the advice based on the predictions
            advice = f"<p style='font-size:24px;'>The highest predicted CPU usage is {max_pred:.2f} at {max_pred_time}. " \
                     f"The lowest predicted CPU usage is {min_pred:.2f} at {min_pred_time}. " \
                     "Administrators should prepare for the highest usage and monitor the system closely during this period.</p>"

            # Display the predictions
            st.subheader('Predicted CPU Usage for the Next 12 Time Points')
            st.write(predicted_df)

            # Plot the predictions
            st.subheader('Prediction Plot')
            fig, ax = plt.subplots(figsize=(18, 9))
            ax.plot(predicted_df.index, predicted_df['predicted_avg_cpu'], label='Predicted CPU Usage', color='orange')
            ax.set_title('Predicted CPU Usage for the Next 12 Time Points', fontsize=24)
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
