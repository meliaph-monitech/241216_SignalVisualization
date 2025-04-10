import streamlit as st
import zipfile
import os
import shutil
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import welch

# Set page layout to wide
st.set_page_config(layout="wide")

# Title
st.title("Laser Welding Signal Visualization with Frequency-Domain Feature Extraction")

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

            # Step 3: Frequency-Domain Features
            st.subheader("Frequency-Domain Features")
            feature_options = [
                "Power in 200-400 Hz Band",
                "Peak Frequency",
                "Spectral Energy",
                "Dominant Frequency Amplitude",
            ]
            selected_feature = st.selectbox("Select Feature to Display:", options=feature_options, index=0)

            # Step 4: Rolling Window
            st.subheader("Rolling Window")
            rolling_window = st.slider("Rolling Window Size (for time-domain to frequency-domain conversion):", min_value=1, max_value=500, value=100)

            # Step 5: Column Selection
            st.subheader("Column Selection")
            selected_columns = st.multiselect(
                "Select Columns to Display:",
                options=column_names,
                default=column_names
            )

            # Step 6: Visualization trigger
            visualize_triggered = st.button("Visualize")

# Helper function for frequency-domain feature extraction
def calculate_frequency_features(data, feature, fs=1000):
    """
    Extract frequency-domain features using Welch's method.
    """
    f, Pxx = welch(data, fs=fs, nperseg=min(len(data), rolling_window))

    if feature == "Power in 200-400 Hz Band":
        band = (f >= 200) & (f <= 400)
        return np.sum(Pxx[band])
    elif feature == "Peak Frequency":
        return f[np.argmax(Pxx)]
    elif feature == "Spectral Energy":
        return np.sum(Pxx)
    elif feature == "Dominant Frequency Amplitude":
        return np.max(Pxx)
    else:
        return None

# Visualization
if uploaded_zip and visualize_triggered:
    # Prepare only the selected columns for visualization
    fig_columns = [go.Figure() for _ in selected_columns]

    for col_idx, column_name in enumerate(selected_columns):
        fig = fig_columns[col_idx]

        for file in filtered_files.keys():
            df = filtered_files[file]
            signal_data = df[column_name]

            # Rolling window to compute frequency-domain features
            feature_data = []
            for start_idx in range(0, len(signal_data) - rolling_window, rolling_window):
                segment = signal_data[start_idx:start_idx + rolling_window]
                feature_value = calculate_frequency_features(segment, selected_feature)
                feature_data.append(feature_value)

            # Add line for this file
            fig.add_trace(go.Scatter(
                x=np.arange(len(feature_data)),
                y=feature_data,
                mode='lines',
                name=file
            ))

        fig.update_layout(
            title=f"Visualization for {column_name} ({selected_feature})",
            xaxis_title="Segment Index",
            yaxis_title="Feature Value",
            height=600,
            showlegend=True,
        )
        st.plotly_chart(fig)
