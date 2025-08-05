import streamlit as st
import pandas as pd
import numpy as np
import zipfile, io, os
import plotly.graph_objects as go
from scipy.signal import savgol_filter, butter, filtfilt

st.set_page_config(layout="wide")

# ---- SESSION STATE INITIALIZATION ----
if "segmented_data" not in st.session_state:
    st.session_state.segmented_data = None
if "observations" not in st.session_state:
    st.session_state.observations = []

# ---- ZIP UPLOAD & EXTRACTION ----
st.sidebar.header("Step 1: Upload & Segmentation")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of CSV files", type="zip")

filter_col = st.sidebar.text_input("Filter Column (for segmentation)")
filter_thresh = st.sidebar.number_input("Filter Threshold", value=0.0)
segment_btn = st.sidebar.button("Bead Segmentation")

# ---- SEGMENTATION LOGIC ----
def bead_segmentation(csv_dict, filter_col, filter_thresh):
    segmented = {}
    for fname, df in csv_dict.items():
        df[filter_col] = pd.to_numeric(df[filter_col], errors='coerce').fillna(0)
        # Example segmentation by threshold crossing:
        bead_id, beads = 1, []
        start_idx = 0
        for i in range(1, len(df)):
            if df[filter_col].iloc[i] < filter_thresh <= df[filter_col].iloc[i-1]:
                beads.append((bead_id, df.iloc[start_idx:i]))
                bead_id += 1
                start_idx = i
        beads.append((bead_id, df.iloc[start_idx:]))  # last bead
        segmented[fname] = {bid: bdf for bid, bdf in beads}
    return segmented

# ---- HANDLE SEGMENTATION ----
if segment_btn and uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        csv_dict = {}
        for fname in z.namelist():
            if fname.endswith(".csv"):
                with z.open(fname) as f:
                    csv_dict[os.path.basename(fname)] = pd.read_csv(f)
    st.session_state.segmented_data = bead_segmentation(csv_dict, filter_col, filter_thresh)
    st.session_state.observations.clear()
    st.success("✅ Bead segmentation complete and locked!")

# ---- ONLY SHOW ANALYSIS AFTER SEGMENTATION ----
if st.session_state.segmented_data:
    st.sidebar.header("Step 2: Observation Control")

    # CSV → Bead dynamic selection
    selected_csv = st.sidebar.selectbox("Select CSV File", list(st.session_state.segmented_data.keys()))
    available_beads = list(st.session_state.segmented_data[selected_csv].keys())
    selected_bead = st.sidebar.selectbox("Select Bead Number", available_beads)

    # Signal column picker
    columns = st.session_state.segmented_data[selected_csv][selected_bead].columns.tolist()
    signal_col = st.sidebar.selectbox("Select Signal Column", columns)

    # OK/NOK classification dropdown
    status = st.sidebar.selectbox("Weld Status", ["OK", "NOK"])

    # Add data button
    if st.sidebar.button("➕ Add Data"):
        new_obs = {
            "csv": selected_csv,
            "bead": selected_bead,
            "signal": signal_col,
            "status": status,
            "data": st.session_state.segmented_data[selected_csv][selected_bead][signal_col].reset_index(drop=True)
        }
        st.session_state.observations.append(new_obs)

    # Reset only analysis (not segmentation)
    if st.sidebar.button("🔄 Reset Analysis (keep segmentation)"):
        st.session_state.observations.clear()

    # ---- PLOTS SECTION ----
    if st.session_state.observations:
        tabs = st.tabs(["Raw Signal", "Smoothed", "Low-pass Filter", "Curve Fit", "FFT Band Intensity"])

        # ---- Raw Signal Plot ----
        with tabs[0]:
            fig = go.Figure()
            for obs in st.session_state.observations:
                color = "green" if obs["status"] == "OK" else "red"
                fig.add_trace(go.Scatter(y=obs["data"], mode="lines", name=f"{obs['csv']} - Bead {obs['bead']} ({obs['status']})", line=dict(color=color)))
            fig.update_layout(title="Raw Signal", xaxis_title="Index", yaxis_title="Signal Value")
            st.plotly_chart(fig, use_container_width=True)

        # ---- Smoothed Signal Plot ----
        with tabs[1]:
            window = st.slider("Savitzky-Golay Window Length", 5, 51, 15, step=2)
            poly = st.slider("Polynomial Order", 2, 5, 2)
            fig = go.Figure()
            for obs in st.session_state.observations:
                smoothed = savgol_filter(obs["data"], window, poly)
                color = "green" if obs["status"] == "OK" else "red"
                fig.add_trace(go.Scatter(y=smoothed, mode="lines", name=f"{obs['csv']} - Bead {obs['bead']}", line=dict(color=color)))
            fig.update_layout(title="Smoothed Signal", xaxis_title="Index", yaxis_title="Signal Value")
            st.plotly_chart(fig, use_container_width=True)

        # ---- Low-pass Filter ----
        with tabs[2]:
            cutoff = st.slider("Low-pass Cutoff Frequency", 0.01, 0.5, 0.1)
            order = st.slider("Filter Order", 1, 5, 2)
            fig = go.Figure()
            for obs in st.session_state.observations:
                b, a = butter(order, cutoff, btype='low', analog=False)
                filtered = filtfilt(b, a, obs["data"])
                color = "green" if obs["status"] == "OK" else "red"
                fig.add_trace(go.Scatter(y=filtered, mode="lines", name=f"{obs['csv']} - Bead {obs['bead']}", line=dict(color=color)))
            fig.update_layout(title="Low-pass Filtered Signal", xaxis_title="Index", yaxis_title="Signal Value")
            st.plotly_chart(fig, use_container_width=True)

        # ---- Curve Fit (Example: Polynomial Fit) ----
        with tabs[3]:
            deg = st.slider("Curve Fit Polynomial Degree", 1, 5, 3)
            fig = go.Figure()
            for obs in st.session_state.observations:
                x = np.arange(len(obs["data"]))
                coeffs = np.polyfit(x, obs["data"], deg)
                fitted = np.polyval(coeffs, x)
                color = "green" if obs["status"] == "OK" else "red"
                fig.add_trace(go.Scatter(y=fitted, mode="lines", name=f"{obs['csv']} - Bead {obs['bead']}", line=dict(color=color)))
            fig.update_layout(title="Curve Fitting", xaxis_title="Index", yaxis_title="Signal Value")
            st.plotly_chart(fig, use_container_width=True)

        # ---- FFT Band Intensity ----
        with tabs[4]:
            band_low, band_high = st.slider("Frequency Band (Hz)", 0.0, 0.5, (0.05, 0.15))
            fig = go.Figure()
            for obs in st.session_state.observations:
                fft_vals = np.fft.rfft(obs["data"])
                freqs = np.fft.rfftfreq(len(obs["data"]))
                band_intensity = np.sum(np.abs(fft_vals[(freqs >= band_low) & (freqs <= band_high)]))
                color = "green" if obs["status"] == "OK" else "red"
                fig.add_trace(go.Bar(x=[f"{obs['csv']} - Bead {obs['bead']}"], y=[band_intensity], name=obs['status'], marker_color=color))
            fig.update_layout(title="Frequency Band Intensity", xaxis_title="Signal", yaxis_title="Intensity")
            st.plotly_chart(fig, use_container_width=True)
