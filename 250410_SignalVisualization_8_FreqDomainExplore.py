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
st.title("Laser Welding Signal Visualization with Frequency Features")

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

            # Step 3: Feature Selection
            st.subheader("Feature Selection")
            feature_options = [
                "Raw Signal",
                "Band Power (200Hz)",
                "Band Power (400Hz)",
                "Spectral Flatness",
                "Peak Frequency (200Hz Region)",
                "Peak Frequency (400Hz Region)",
                "Band Power Ratio (200Hz/Total)",
                "Band Power Ratio (400Hz/Total)"
            ]
            selected_feature = st.selectbox("Select Feature to Display:", options=feature_options, index=0)

            # Step 4: Rolling Window
            st.subheader("Rolling Window")
            rolling_window = st.slider("Rolling Window Size:", min_value=1, max_value=500, value=50)

            # Step 5: Bead Selection
            st.subheader("Bead Selection")
            bead_input = st.text_input("Enter Bead Numbers to Visualize (default Bead No.1, blank for all):", value="1")
            bead_numbers = [int(b.strip()) for b in bead_input.split(',') if b.strip().isdigit()] if bead_input else None

            # Step 6: Column Selection
            st.subheader("Column Selection")
            selected_columns = st.multiselect(
                "Select Columns to Display:",
                options=column_names,
                default=column_names
            )

            # Step 7: Normalization Toggle
            st.subheader("Normalization")
            normalize_data = st.checkbox("Normalize data per chosen bead number", value=True)

            # Step 8: Visualization trigger
            visualize_triggered = st.button("Visualize")

# Helper functions for feature extraction
def calculate_frequency_features(data, feature, fs=1000):
    # Welch method parameters
    nperseg = 256

    if feature == "Raw Signal":
        return data
    elif feature == "Band Power (200Hz)":
        def band_power(segment):
            f, Pxx = welch(segment, fs=fs, nperseg=nperseg)
            band = (f >= 190) & (f <= 210)  # 200Hz band
            return np.sum(Pxx[band])
        return data.rolling(rolling_window).apply(band_power, raw=False)
    elif feature == "Band Power (400Hz)":
        def band_power(segment):
            f, Pxx = welch(segment, fs=fs, nperseg=nperseg)
            band = (f >= 390) & (f <= 410)  # 400Hz band
            return np.sum(Pxx[band])
        return data.rolling(rolling_window).apply(band_power, raw=False)
    elif feature == "Spectral Flatness":
        def spectral_flatness(segment):
            f, Pxx = welch(segment, fs=fs, nperseg=nperseg)
            geometric_mean = np.exp(np.mean(np.log(Pxx + 1e-12)))  # Avoid log(0)
            arithmetic_mean = np.mean(Pxx)
            return geometric_mean / arithmetic_mean
        return data.rolling(rolling_window).apply(spectral_flatness, raw=False)
    elif feature == "Peak Frequency (200Hz Region)":
        def peak_frequency(segment):
            f, Pxx = welch(segment, fs=fs, nperseg=nperseg)
            band = (f >= 190) & (f <= 210)
            return f[band][np.argmax(Pxx[band])]
        return data.rolling(rolling_window).apply(peak_frequency, raw=False)
    elif feature == "Peak Frequency (400Hz Region)":
        def peak_frequency(segment):
            f, Pxx = welch(segment, fs=fs, nperseg=nperseg)
            band = (f >= 390) & (f <= 410)
            return f[band][np.argmax(Pxx[band])]
        return data.rolling(rolling_window).apply(peak_frequency, raw=False)
    elif feature == "Band Power Ratio (200Hz/Total)":
        def band_ratio(segment):
            f, Pxx = welch(segment, fs=fs, nperseg=nperseg)
            total_power = np.sum(Pxx)
            band_power = np.sum(Pxx[(f >= 190) & (f <= 210)])
            return band_power / total_power
        return data.rolling(rolling_window).apply(band_ratio, raw=False)
    elif feature == "Band Power Ratio (400Hz/Total)":
        def band_ratio(segment):
            f, Pxx = welch(segment, fs=fs, nperseg=nperseg)
            total_power = np.sum(Pxx)
            band_power = np.sum(Pxx[(f >= 390) & (f <= 410)])
            return band_power / total_power
        return data.rolling(rolling_window).apply(band_ratio, raw=False)

# Visualization
if uploaded_zip and visualize_triggered:
    fig_columns = [go.Figure() for _ in selected_columns]

    for col_idx, column_name in enumerate(selected_columns):
        fig = fig_columns[col_idx]

        for file in filtered_files.keys():
            df = filtered_files[file]
            signal_data = df[column_name]

            # Normalize if selected
            if normalize_data:
                signal_data = (signal_data - signal_data.min()) / (signal_data.max() - signal_data.min())

            # Compute features
            feature_data = calculate_frequency_features(signal_data, selected_feature)

            # Add line for this file
            fig.add_trace(go.Scatter(
                x=np.arange(len(feature_data)),
                y=feature_data,
                mode='lines',
                name=file
            ))

        fig.update_layout(
            title=f"Visualization for {column_name} ({selected_feature})",
            xaxis_title="Index",
            yaxis_title="Values",
            height=600,
            showlegend=True,
        )
        st.plotly_chart(fig)
