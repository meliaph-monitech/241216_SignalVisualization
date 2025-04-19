import streamlit as st
import zipfile
import os
import shutil
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import scipy.signal as signal

# Set page layout to wide
st.set_page_config(layout="wide")

# Streamlit Title
st.title("Laser Welding Signal Visualization with Frequency-Based Rolling Variance")

# Sidebar for inputs
with st.sidebar:
    st.header("Input Parameters")

    # Step 1: Upload ZIP File
    uploaded_zip = st.file_uploader("Upload a ZIP file containing CSV files:", type="zip")

    if uploaded_zip:
        extract_dir = "extracted_csvs"
        if os.path.exists(extract_dir):
            try:
                # Safely remove the existing extraction directory and its contents
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

            # Step 4: Frequency Input
            st.subheader("Frequency Selection")
            chosen_frequency = st.number_input("Enter the frequency (Hz) to analyze:", min_value=1, value=240)

            # Step 5: Rolling Window
            st.subheader("Rolling Window")
            rolling_window = st.slider("Rolling Window Size:", min_value=1, max_value=500, value=50)

            # Step 6: Normalization Toggle
            st.subheader("Normalization")
            normalize_data = st.checkbox("Normalize data per chosen frequency", value=True)

            # Step 7: Visualization trigger
            visualize_triggered = st.button("Visualize")

if uploaded_zip and visualize_triggered:
    # Prepare for signal intensity extraction
    file_colors = {file: f"rgb({(hash(file) % 256)},{(hash(file + 'g') % 256)},{(hash(file + 'b') % 256)})" for file in selected_files}
    
    # Initialize plotly figure
    fig = go.Figure()

    for file in selected_files:
        df = filtered_files[file]

        # Extract the raw signal values
        signal_column = df.iloc[:, 0].values  # Assuming the first column contains the raw signal
        time_column = np.arange(len(signal_column)) / 10000  # Assuming a sampling rate of 10 kHz
        
        # Compute the spectrogram
        fs = 10000  # Sampling frequency
        f, t, Sxx = signal.spectrogram(signal_column, fs=fs, nperseg=1024, noverlap=512)
        
        # Find the closest frequency index to the chosen frequency
        freq_idx = np.abs(f - chosen_frequency).argmin()
        signal_intensity = Sxx[freq_idx, :]  # Extract signal intensity at the chosen frequency

        # Normalize the signal intensity if selected
        if normalize_data:
            signal_intensity = (signal_intensity - np.min(signal_intensity)) / (np.max(signal_intensity) - np.min(signal_intensity))

        # Calculate rolling variance
        rolling_variance = pd.Series(signal_intensity).rolling(rolling_window, min_periods=1).var()

        # Add the rolling variance to the plot
        fig.add_trace(go.Scatter(
            x=t,
            y=rolling_variance,
            mode='lines',
            name=f"Rolling Variance - {file}",
            line=dict(color=file_colors[file], width=2)
        ))

    # Update plot layout
    fig.update_layout(
        title="Rolling Variance of Signal Intensity at Chosen Frequency",
        xaxis_title="Time (s)",
        yaxis_title="Rolling Variance",
        height=600,
        showlegend=True
    )

    # Display the plot
    st.plotly_chart(fig)
