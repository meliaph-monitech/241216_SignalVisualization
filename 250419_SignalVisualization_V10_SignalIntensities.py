import streamlit as st
import zipfile
import os
import shutil
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import spectrogram

# Set page layout to wide
st.set_page_config(layout="wide")

st.title("Signal Intensities with Rolling Limits Visualization")

# Sidebar for inputs
with st.sidebar:
    st.header("Input Parameters")

    # Step 1: Upload ZIP file
    uploaded_zip = st.file_uploader("Upload a ZIP file containing CSV files:", type="zip")

    if uploaded_zip:
        extract_dir = "extracted_csvs"
        if os.path.exists(extract_dir):
            try:
                shutil.rmtree(extract_dir)  # Remove the existing directory
            except Exception as e:
                st.error(f"Error cleaning up previous files: {e}")

        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]

        if csv_files:
            st.success(f"Extracted {len(csv_files)} CSV files.")

            # Step 2: Select processing parameters
            st.subheader("Signal Processing Parameters")
            sample_df = pd.read_csv(os.path.join(extract_dir, csv_files[0]))
            column_names = sample_df.columns.tolist()

            filter_column = st.selectbox("Select the Filter Column:", column_names)
            filter_threshold = st.number_input("Set Filter Threshold:", value=1.0)

            # Frequency selection
            sampling_frequency = st.number_input("Sampling Frequency (Hz):", min_value=1000, max_value=50000, value=10000)
            chosen_frequency = st.number_input("Frequency of Interest (Hz):", min_value=1, value=240)

            # Rolling window for limits
            rolling_window = st.slider("Rolling Window Size:", min_value=1, max_value=500, value=50)

            # Visualization trigger
            visualize_triggered = st.button("Visualize")

if uploaded_zip and visualize_triggered:
    # Dictionary to store signal intensities for each CSV file
    signal_intensities = {}

    for file in csv_files:
        df = pd.read_csv(os.path.join(extract_dir, file))

        # Filter rows based on the threshold
        filtered_df = df[df[filter_column] > filter_threshold]

        if filtered_df.empty:
            continue

        # Calculate spectrogram for the filtered column
        f, t, Sxx = spectrogram(
            filtered_df[filter_column].to_numpy(),
            fs=sampling_frequency,
            nperseg=min(1024, len(filtered_df)),
            noverlap=256,
            nfft=2048
        )
        Sxx_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)

        # Extract signal intensity at the chosen frequency range
        freq_indices = np.where((f >= chosen_frequency - 5) & (f <= chosen_frequency + 5))[0]
        if len(freq_indices) > 0:
            intensity = np.mean(Sxx_dB[freq_indices, :], axis=0)
            signal_intensities[file] = (t, intensity)

    # Combine all signal intensities for rolling limit calculation
    all_intensities = [intensity for _, intensity in signal_intensities.values()]
    min_length = min(map(len, all_intensities))
    truncated_intensities = [intensity[:min_length] for intensity in all_intensities]
    stacked_intensities = np.vstack(truncated_intensities)

    # Calculate rolling limits (mean ± 2*std)
    mean_intensity = np.mean(stacked_intensities, axis=0)
    std_intensity = np.std(stacked_intensities, axis=0)
    upper_limit = pd.Series(mean_intensity + 2 * std_intensity).rolling(rolling_window, min_periods=1).mean()
    lower_limit = pd.Series(mean_intensity - 2 * std_intensity).rolling(rolling_window, min_periods=1).mean()

    # Visualization
    fig = go.Figure()

    for file, (t, intensity) in signal_intensities.items():
        fig.add_trace(go.Scatter(
            x=t[:len(intensity)],
            y=intensity[:len(t)],
            mode='lines',
            name=file,
            line=dict(width=1),
            hoverinfo='text',
            text=[f"File: {file}<br>Time: {time:.2f}s<br>Intensity: {val:.2f}dB"
                  for time, val in zip(t, intensity)]
        ))

    # Add rolling upper and lower limits
    fig.add_trace(go.Scatter(
        x=t[:len(upper_limit)],
        y=upper_limit,
        mode='lines',
        name="Upper Limit",
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=t[:len(lower_limit)],
        y=lower_limit,
        mode='lines',
        name="Lower Limit",
        line=dict(color='blue', width=2, dash='dash')
    ))

    # Configure layout
    fig.update_layout(
        title="Signal Intensities Over Time with Rolling Limits",
        xaxis_title="Time (s)",
        yaxis_title="Signal Intensity (dB)",
        height=600,
        showlegend=True
    )

    st.plotly_chart(fig)
