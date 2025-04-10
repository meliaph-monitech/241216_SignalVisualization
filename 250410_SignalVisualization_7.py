import streamlit as st
import zipfile
import os
import shutil
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
from scipy.signal import welch

# Set page layout to wide
st.set_page_config(layout="wide")

# Title
st.title("Laser Welding Signal Visualization with Feature Extraction")

# Sidebar for inputs
with st.sidebar:
    st.header("Input Parameters")

    # Step 1: Upload ZIP File
    uploaded_zip = st.file_uploader("Upload a ZIP file containing CSV files:", type="zip")

    if uploaded_zip:
        extract_dir = "extracted_csvs"
        if os.path.exists(extract_dir):
            try:
                shutil.rmtree(extract_dir)
            except Exception as e:
                st.error(f"Error cleaning up previous files: {e}")

        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]

        if csv_files:
            st.success(f"Extracted {len(csv_files)} CSV files.")

            # Step 2: Filter Data
            st.subheader("Filter Data")
            sample_df = pd.read_csv(os.path.join(extract_dir, csv_files[0]))
            column_names = sample_df.columns.tolist()

            filter_column = st.selectbox("Select the Filter Column to reduce data:", column_names)
            filter_threshold = st.number_input("Set Filter Threshold:", value=1.0)

            # Filter CSV files based on the threshold
            filtered_files = {}
            for file in csv_files:
                df = pd.read_csv(os.path.join(extract_dir, file))
                if (df[filter_column] > filter_threshold).any():
                    filtered_files[file] = df

            st.success(f"Filtered down to {len(filtered_files)} files after applying the threshold.")

            # Step 3: Select CSV File
            st.subheader("Select CSV File")
            file_list = list(filtered_files.keys())
            selected_files = st.multiselect("Select CSV file(s) to visualize:", options=["All"] + file_list, default="All")
            if "All" in selected_files:
                selected_files = file_list

            # Step 4: Feature Selection
            st.subheader("Feature Selection")
            feature_options = [
                "Raw Signal",
                "Mean",
                "RMS",
                "Spectral Entropy",
                "Peak Frequency",
                "Bandwidth",
                "Skewness",
                "Kurtosis",
                "Band Power",
                "Max Amplitude",
                "Sum of Amplitudes",
                "Moving Average"
            ]
            selected_feature = st.selectbox("Select Feature to Display:", options=feature_options, index=0)

            # Step 5: Rolling Window
            st.subheader("Rolling Window")
            rolling_window = st.slider("Rolling Window Size:", min_value=1, max_value=500, value=50)

            # Step 6: Visualization trigger
            visualize_triggered = st.button("Visualize")

# Helper functions for feature extraction
def calculate_features(data, feature, fs=1000):
    if feature == "Raw Signal":
        return data
    elif feature == "Mean":
        return data.rolling(rolling_window).mean()
    elif feature == "RMS":
        return np.sqrt(data.rolling(rolling_window).mean() ** 2)
    elif feature == "Spectral Entropy":
        def spectral_entropy(segment):
            f, Pxx = welch(segment, fs=fs)
            Pxx = Pxx / np.sum(Pxx)  # Normalize power spectrum
            return -np.sum(Pxx * np.log2(Pxx + 1e-12))  # Avoid log(0)
        return data.rolling(rolling_window).apply(spectral_entropy, raw=False)
    elif feature == "Peak Frequency":
        def peak_frequency(segment):
            f, Pxx = welch(segment, fs=fs)
            return f[np.argmax(Pxx)]
        return data.rolling(rolling_window).apply(peak_frequency, raw=False)
    elif feature == "Bandwidth":
        def bandwidth(segment):
            f, Pxx = welch(segment, fs=fs)
            return np.sqrt(np.sum(Pxx * (f - np.mean(f)) ** 2))
        return data.rolling(rolling_window).apply(bandwidth, raw=False)
    elif feature == "Skewness":
        return data.rolling(rolling_window).apply(skew, raw=True)
    elif feature == "Kurtosis":
        return data.rolling(rolling_window).apply(kurtosis, raw=True)
    elif feature == "Band Power":
        def band_power(segment):
            f, Pxx = welch(segment, fs=fs)
            band = (f >= 200) & (f <= 400)  # Example: 200Hz to 400Hz
            return np.sum(Pxx[band])
        return data.rolling(rolling_window).apply(band_power, raw=False)
    elif feature == "Max Amplitude":
        return data.rolling(rolling_window).max()
    elif feature == "Sum of Amplitudes":
        return data.rolling(rolling_window).sum()
    elif feature == "Moving Average":
        return data.rolling(rolling_window).mean()

# Visualization
if uploaded_zip and visualize_triggered:
    for file in selected_files:
        df = filtered_files[file]
        fig = go.Figure()
        for col in df.columns[:3]:  # Assuming first three columns are signals
            signal_data = df[col]
            feature_data = calculate_features(signal_data, selected_feature)

            fig.add_trace(go.Scatter(
                x=np.arange(len(feature_data)),
                y=feature_data,
                mode='lines',
                name=f"{col} ({selected_feature})"
            ))

        fig.update_layout(
            title=f"Visualization for {file}",
            xaxis_title="Index",
            yaxis_title="Values",
            height=600
        )
        st.plotly_chart(fig)
