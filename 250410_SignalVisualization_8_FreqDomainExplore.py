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
st.title("Laser Welding Signal Visualization with Feature Extraction and Frequency Analysis")

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
                "Moving Average",
                "Frequency Spectrum"  # New option for frequency spectrum
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
    elif feature == "Frequency Spectrum":  # New feature for frequency spectrum
        def frequency_spectrum(segment):
            fft_data = np.fft.fft(segment)
            freqs = np.fft.fftfreq(len(segment), d=1/fs)
            return abs(fft_data[:len(freqs) // 2])  # Only return positive frequencies
        return data.rolling(rolling_window).apply(frequency_spectrum, raw=False)

# Visualization
if uploaded_zip and visualize_triggered:
    # Prepare only the selected columns for visualization
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
            feature_data = calculate_features(signal_data, selected_feature)

            # Add line for this file
            if selected_feature == "Frequency Spectrum":  # Special handling for frequency spectrum
                freqs = np.fft.fftfreq(len(signal_data), d=1 / 1000)
                positive_freqs = freqs[:len(freqs) // 2]
                fig.add_trace(go.Scatter(
                    x=positive_freqs,
                    y=feature_data.iloc[len(feature_data) // 2].values,  # Example: take the middle segment
                    mode='lines',
                    name=file
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=np.arange(len(feature_data)),
                    y=feature_data,
                    mode='lines',
                    name=file
                ))

        fig.update_layout(
            title=f"Visualization for {column_name} ({selected_feature})",
            xaxis_title="Frequency (Hz)" if selected_feature == "Frequency Spectrum" else "Index",
            yaxis_title="Amplitude" if selected_feature == "Frequency Spectrum" else "Values",
            height=600,
            showlegend=True,
        )
        st.plotly_chart(fig)
