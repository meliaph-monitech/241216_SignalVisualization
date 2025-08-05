import streamlit as st
import pandas as pd
import numpy as np
import zipfile, io, os
import plotly.graph_objects as go
from scipy.signal import savgol_filter, butter, filtfilt

st.set_page_config(layout="wide")

# --- Utility: Bead Segmentation ---
def segment_beads(df, column, threshold):
    start_indices, end_indices = [], []
    signal = df[column].to_numpy()
    i = 0
    while i < len(signal):
        if signal[i] > threshold:
            start = i
            while i < len(signal) and signal[i] > threshold:
                i += 1
            end = i - 1
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    return list(zip(start_indices, end_indices))

# --- Session State ---
if "segmented_data" not in st.session_state:
    st.session_state.segmented_data = None
if "observations" not in st.session_state:
    st.session_state.observations = []

# --- Sidebar: Upload & Segmentation ---
st.sidebar.header("Step 1: Upload & Segmentation")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of CSV files", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        first_csv = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(first_csv) as f:
            sample_df = pd.read_csv(f)
    columns = sample_df.columns.tolist()
    seg_col = st.sidebar.selectbox("Column for Segmentation", columns)
    seg_thresh = st.sidebar.number_input("Segmentation Threshold", value=1.0)
    segment_btn = st.sidebar.button("Bead Segmentation")

# --- Perform Segmentation ---
if uploaded_zip and 'segment_btn' in locals() and segment_btn:
    segmented_data = {}
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.csv'):
                with zip_ref.open(file_name) as f:
                    df = pd.read_csv(f)
                bead_ranges = segment_beads(df, seg_col, seg_thresh)
                bead_dict = {}
                for idx, (start, end) in enumerate(bead_ranges, start=1):
                    bead_dict[idx] = df.iloc[start:end+1].reset_index(drop=True)
                segmented_data[os.path.basename(file_name)] = bead_dict
    st.session_state.segmented_data = segmented_data
    st.session_state.observations.clear()
    st.success("✅ Bead segmentation complete and locked!")

# --- Analysis UI After Segmentation ---
if st.session_state.segmented_data:
    st.sidebar.header("Step 2: Add Data for Analysis")
    selected_csv = st.sidebar.selectbox("Select CSV File", list(st.session_state.segmented_data.keys()))
    available_beads = list(st.session_state.segmented_data[selected_csv].keys())
    selected_bead = st.sidebar.selectbox("Select Bead Number", available_beads)

    # Pick signal column
    bead_df = st.session_state.segmented_data[selected_csv][selected_bead]
    signal_col = st.sidebar.selectbox("Select Signal Column", bead_df.columns.tolist())

    # Weld Status (OK/NOK)
    status = st.sidebar.selectbox("Weld Status", ["OK", "NOK"])

    # Add data button
    if st.sidebar.button("➕ Add Data"):
        st.session_state.observations.append({
            "csv": selected_csv,
            "bead": selected_bead,
            "signal": signal_col,
            "status": status,
            "data": bead_df[signal_col].reset_index(drop=True)
        })

    # Reset only analysis
    if st.sidebar.button("🔄 Reset Analysis (keep segmentation)"):
        st.session_state.observations.clear()

    # --- Visualization Tabs ---
    if st.session_state.observations:
        tabs = st.tabs(["Raw Signal", "Smoothed", "Low-pass Filter", "Curve Fit", "FFT Band Intensity"])

        # --- Raw Signal ---
        with tabs[0]:
            fig = go.Figure()
            for obs in st.session_state.observations:
                color = "green" if obs["status"] == "OK" else "red"
                fig.add_trace(go.Scatter(y=obs["data"], mode="lines",
                                         name=f"{obs['csv']} - Bead {obs['bead']} ({obs['status']})",
                                         line=dict(color=color)))
            fig.update_layout(title="Raw Signal", xaxis_title="Index", yaxis_title="Signal Value")
            st.plotly_chart(fig, use_container_width=True)

        # --- Smoothed Signal ---
        with tabs[1]:
            window = st.slider("Savitzky-Golay Window Length", 5, 51, 15, step=2)
            poly = st.slider("Polynomial Order", 2, 5, 2)
            fig = go.Figure()
            for obs in st.session_state.observations:
                smoothed = savgol_filter(obs["data"], window, poly)
                color = "green" if obs["status"] == "OK" else "red"
                fig.add_trace(go.Scatter(y=smoothed, mode="lines",
                                         name=f"{obs['csv']} - Bead {obs['bead']} ({obs['status']})",
                                         line=dict(color=color)))
            fig.update_layout(title="Smoothed Signal", xaxis_title="Index", yaxis_title="Signal Value")
            st.plotly_chart(fig, use_container_width=True)

        # --- Low-pass Filter ---
        with tabs[2]:
            cutoff = st.slider("Low-pass Cutoff Frequency", 0.01, 0.5, 0.1)
            order = st.slider("Filter Order", 1, 5, 2)
            fig = go.Figure()
            for obs in st.session_state.observations:
                b, a = butter(order, cutoff, btype='low', analog=False)
                filtered = filtfilt(b, a, obs["data"])
                color = "green" if obs["status"] == "OK" else "red"
                fig.add_trace(go.Scatter(y=filtered, mode="lines",
                                         name=f"{obs['csv']} - Bead {obs['bead']} ({obs['status']})",
                                         line=dict(color=color)))
            fig.update_layout(title="Low-pass Filtered Signal", xaxis_title="Index", yaxis_title="Signal Value")
            st.plotly_chart(fig, use_container_width=True)

        # --- Curve Fit ---
        with tabs[3]:
            deg = st.slider("Curve Fit Polynomial Degree", 1, 100, 5)
            fig = go.Figure()
            for obs in st.session_state.observations:
                x = np.arange(len(obs["data"]))
                coeffs = np.polyfit(x, obs["data"], deg)
                fitted = np.polyval(coeffs, x)
                color = "green" if obs["status"] == "OK" else "red"
                fig.add_trace(go.Scatter(y=fitted, mode="lines",
                                         name=f"{obs['csv']} - Bead {obs['bead']} ({obs['status']})",
                                         line=dict(color=color)))
            fig.update_layout(title="Curve Fit Signal", xaxis_title="Index", yaxis_title="Signal Value")
            st.plotly_chart(fig, use_container_width=True)

        # # --- FFT Band Intensity ---
        # with tabs[4]:
        #     # Sampling rate input (needed for correct FFT frequency scaling)
        #     sampling_rate = st.number_input("Sampling Rate (Hz)", value=2000, min_value=1)
        
        #     # Frequency band slider (0-1000Hz in 50Hz steps)
        #     band_low, band_high = st.slider("Frequency Band (Hz)", 
        #                                     min_value=0, 
        #                                     max_value=1000, 
        #                                     value=(50, 150), 
        #                                     step=50)
            
        #     fig = go.Figure()
        #     intensities = []
        #     labels = []
            
        #     for obs in st.session_state.observations:
        #         fft_vals = np.fft.rfft(obs["data"])
        #         freqs = np.fft.rfftfreq(len(obs["data"]), d=1.0/sampling_rate)  # Convert to Hz
        #         mask = (freqs >= band_low) & (freqs <= band_high)
        #         band_intensity = np.sum(np.abs(fft_vals[mask]))
        #         intensities.append(band_intensity)
        #         labels.append(f"{obs['csv']} - Bead {obs['bead']} ({obs['status']})")
        #         color = "green" if obs["status"] == "OK" else "red"
        #         fig.add_trace(go.Bar(x=[labels[-1]], y=[band_intensity], marker_color=color, name=obs['status']))
        
        #     # Auto-scale y-axis based on max intensity
        #     if intensities:
        #         fig.update_yaxes(range=[0, max(intensities)*1.2])  # add 20% padding
            
        #     fig.update_layout(title="FFT Band Intensity", xaxis_title="Signal", yaxis_title="Intensity")
        #     st.plotly_chart(fig, use_container_width=True)

        # --- FFT Spectrum (Zoomed to Selected Band) ---
        with tabs[4]:
            sampling_rate = st.number_input("Sampling Rate (Hz)", value=2000, min_value=1)
        
            # Frequency band slider (0-1000Hz in 50Hz steps)
            band_low, band_high = st.slider("Frequency Band (Hz)",
                                            min_value=0,
                                            max_value=1000,
                                            value=(50, 150),
                                            step=50)
        
            fig = go.Figure()
        
            for obs in st.session_state.observations:
                fft_vals = np.fft.rfft(obs["data"])
                freqs = np.fft.rfftfreq(len(obs["data"]), d=1.0/sampling_rate)
                magnitude = np.abs(fft_vals)
        
                # Filter to zoom only the selected frequency band
                mask = (freqs >= band_low) & (freqs <= band_high)
                freqs_zoom = freqs[mask]
                magnitude_zoom = magnitude[mask]
        
                color = "green" if obs["status"] == "OK" else "red"
                fig.add_trace(go.Scatter(x=freqs_zoom, y=magnitude_zoom,
                                         mode="lines",
                                         name=f"{obs['csv']} - Bead {obs['bead']} ({obs['status']})",
                                         line=dict(color=color)))
        
            # Update layout for zoomed view
            fig.update_layout(
                title=f"FFT Spectrum (Zoomed {band_low}-{band_high} Hz)",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                xaxis=dict(range=[band_low, band_high])
            )
            st.plotly_chart(fig, use_container_width=True)
