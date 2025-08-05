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

def aggregate_for_step(x, y, interval):
    """Aggregate x and y into buckets of given interval for step plotting."""
    agg_x = x[::interval]
    agg_y = [np.mean(y[i:i+interval]) for i in range(0, len(y), interval)]
    return agg_x[:len(agg_y)], agg_y

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
        tabs = st.tabs([
            "Raw Signal", 
            "Smoothed", 
            "Low-pass Filter", 
            "Curve Fit", 
            "FFT Band Intensity", 
            "Signal Intensity (dB)"  # NEW TAB
        ])

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
            use_step = st.checkbox("Display as Step Line (Smoothed)", value=False)
            step_interval = st.slider("Step Interval (points) - Smoothed", 10, 500, 50) if use_step else None
        
            fig = go.Figure()
            for obs in st.session_state.observations:
                smoothed = savgol_filter(obs["data"], window, poly)
                x_vals = np.arange(len(smoothed))
                if use_step:
                    x_vals, smoothed = aggregate_for_step(x_vals, smoothed, step_interval)
                color = "green" if obs["status"] == "OK" else "red"
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=smoothed,
                    mode="lines",
                    name=f"{obs['csv']} - Bead {obs['bead']} ({obs['status']})",
                    line=dict(color=color, shape="hv" if use_step else "linear")
                ))
            fig.update_layout(title="Smoothed Signal", xaxis_title="Index", yaxis_title="Signal Value")
            st.plotly_chart(fig, use_container_width=True)
        
        # --- Low-pass Filter ---
        with tabs[2]:
            cutoff = st.slider("Low-pass Cutoff Frequency", 0.01, 0.5, 0.1)
            order = st.slider("Filter Order", 1, 5, 2)
            use_step = st.checkbox("Display as Step Line (Low-pass)", value=False)
            step_interval = st.slider("Step Interval (points) - Low-pass", 10, 500, 50) if use_step else None
        
            fig = go.Figure()
            for obs in st.session_state.observations:
                b, a = butter(order, cutoff, btype='low', analog=False)
                filtered = filtfilt(b, a, obs["data"])
                x_vals = np.arange(len(filtered))
                if use_step:
                    x_vals, filtered = aggregate_for_step(x_vals, filtered, step_interval)
                color = "green" if obs["status"] == "OK" else "red"
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=filtered,
                    mode="lines",
                    name=f"{obs['csv']} - Bead {obs['bead']} ({obs['status']})",
                    line=dict(color=color, shape="hv" if use_step else "linear")
                ))
            fig.update_layout(title="Low-pass Filtered Signal", xaxis_title="Index", yaxis_title="Signal Value")
            st.plotly_chart(fig, use_container_width=True)
        
        # --- Curve Fit ---
        with tabs[3]:
            deg = st.slider("Curve Fit Polynomial Degree", 1, 100, 5)
            use_step = st.checkbox("Display as Step Line (Curve Fit)", value=False)
            step_interval = st.slider("Step Interval (points) - Curve Fit", 10, 500, 50) if use_step else None

            fig = go.Figure()
            for obs in st.session_state.observations:
                x = np.arange(len(obs["data"]))
                coeffs = np.polyfit(x, obs["data"], deg)
                fitted = np.polyval(coeffs, x)
                if use_step:
                    x, fitted = aggregate_for_step(x, fitted, step_interval)
                color = "green" if obs["status"] == "OK" else "red"
                fig.add_trace(go.Scatter(
                    x=x,
                    y=fitted,
                    mode="lines",
                    name=f"{obs['csv']} - Bead {obs['bead']} ({obs['status']})",
                    line=dict(color=color, shape="hv" if use_step else "linear")
                ))
            fig.update_layout(title="Curve Fit Signal", xaxis_title="Index", yaxis_title="Signal Value")
            st.plotly_chart(fig, use_container_width=True)

        # --- FFT Spectrum (Zoomed to Selected Band, in dB, Reversed) ---
        with tabs[4]:
            sampling_rate = st.number_input("Sampling Rate (Hz) - Spectrum", value=2000, min_value=1)
        
            band_low, band_high = st.slider("Frequency Band (Hz) - Spectrum",
                                            min_value=0,
                                            max_value=1000,
                                            value=(50, 150),
                                            step=50)
        
            fig = go.Figure()
        
            for obs in st.session_state.observations:
                fft_vals = np.fft.rfft(obs["data"])
                freqs = np.fft.rfftfreq(len(obs["data"]), d=1.0/sampling_rate)
                magnitude = np.abs(fft_vals)
                magnitude_db = 20 * np.log10(magnitude + 1e-12)  # Convert to dB
        
                mask = (freqs >= band_low) & (freqs <= band_high)
                freqs_zoom = freqs[mask]
                magnitude_zoom = magnitude_db[mask]
        
                color = "green" if obs["status"] == "OK" else "red"
                fig.add_trace(go.Scatter(
                    x=freqs_zoom,
                    y=magnitude_zoom,
                    mode="lines",
                    name=f"{obs['csv']} - Bead {obs['bead']} ({obs['status']})",
                    line=dict(color=color),
                    fill='tozeroy',  # Fill below the curve
                    fillcolor='rgba(0,0,0,0.05)'
                ))
        
            fig.update_layout(
                title=f"FFT Spectrum (Zoomed {band_low}-{band_high} Hz, in dB)",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Signal Intensity (dB)",
                xaxis=dict(range=[band_low, band_high]),
                yaxis=dict(autorange="reversed")  # Flip Y-axis
            )
        
            st.plotly_chart(fig, use_container_width=True)

        # --- Signal Intensity (dB) ---
        with tabs[5]:
            st.subheader("Signal Intensity (dB) Over Time")
            
            sampling_rate = st.number_input("Sampling Rate (Hz) - Intensity", value=10000, min_value=1)
        
            # Checkbox to toggle between band or exact frequency
            use_exact_freq = st.checkbox("Use Exact Frequency (Hz) Instead of Band")
        
            if use_exact_freq:
                exact_freq = st.number_input("Exact Frequency (Hz)", value=300, min_value=0, max_value=1000, step=50)
            else:
                band_low, band_high = st.slider("Frequency Band (Hz) - Intensity", 0, 1000, (130, 170), step=10)
        
            window_size = st.slider("Window Size (ms) - Intensity", 1, 500, 50)  # sliding FFT window
            overlap = st.slider("Window Overlap (%) - Intensity", 0, 99, 99)
        
            fig = go.Figure()
        
            for obs in st.session_state.observations:
                data = obs["data"].to_numpy()
                window_points = int((window_size / 1000) * sampling_rate)
                step_points = int(window_points * (1 - overlap / 100))
                times, intensities = [], []
        
                for start in range(0, len(data) - window_points, step_points):
                    segment = data[start:start + window_points]
                    fft_vals = np.fft.rfft(segment)
                    freqs = np.fft.rfftfreq(len(segment), d=1.0/sampling_rate)
        
                    if use_exact_freq:
                        # Find closest FFT bin to the exact frequency
                        idx = (np.abs(freqs - exact_freq)).argmin()
                        band_mag = np.abs(fft_vals[idx])
                    else:
                        # Use frequency band range
                        mask = (freqs >= band_low) & (freqs <= band_high)
                        band_mag = np.sum(np.abs(fft_vals[mask]))
        
                    intensity_db = 20 * np.log10(band_mag + 1e-12)  # avoid log(0)
                    times.append(start / sampling_rate)  # time in seconds
                    intensities.append(intensity_db)
        
                color = "green" if obs["status"] == "OK" else "red"
                fig.add_trace(go.Scatter(
                    x=times,
                    y=intensities,
                    mode="lines",
                    name=f"{obs['csv']} - Bead {obs['bead']} ({obs['status']})",
                    line=dict(color=color)
                ))
        
            fig.update_layout(
                title="Signal Intensity (dB) Over Time",
                xaxis_title="Time (seconds)",
                yaxis_title="Intensity (dB)"
            )
            st.plotly_chart(fig, use_container_width=True)
