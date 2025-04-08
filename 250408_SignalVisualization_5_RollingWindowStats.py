import streamlit as st
import zipfile
import os
import shutil
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Set page layout to wide
st.set_page_config(layout="wide")

# Streamlit Title
st.title("Laser Welding Signal Visualization")

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

            # Step 4: Bead Selection
            st.subheader("Bead Selection")
            bead_input = st.text_input("Enter Bead Numbers to Visualize (default Bead No.1, blank for all):", value="1")
            bead_numbers = [int(b.strip()) for b in bead_input.split(',') if b.strip().isdigit()] if bead_input else None

            # Step 5: Limit Calculation Method
            st.subheader("Limit Calculation Method")
            method = st.selectbox("Select Method for Limit Calculation:", ["Standard Deviation", "Percentile"])

            # Step 6: Rolling Window
            st.subheader("Rolling Window")
            rolling_window = st.slider("Rolling Window Size:", min_value=1, max_value=500, value=50)

            # Step 7: Visualization trigger
            visualize_triggered = st.button("Visualize")

if uploaded_zip and visualize_triggered:
    bead_data = {col_idx: [] for col_idx in range(3)}
    bead_limits = {col_idx: {} for col_idx in range(3)}
    file_colors = {file: f"rgb({(hash(file) % 256)},{(hash(file + 'g') % 256)},{(hash(file + 'b') % 256)})" for file in selected_files}

    bead_segments_by_index = {col_idx: {} for col_idx in range(3)}

    for file in selected_files:
        df = filtered_files[file]
        filter_values = df[filter_column].to_numpy()
        start_points = []
        end_points = []
        i = 0
        while i < len(filter_values):
            if filter_values[i] > filter_threshold:
                if not end_points or i > end_points[-1]:
                    start_points.append(i)
                while i < len(filter_values) and filter_values[i] > filter_threshold:
                    i += 1
                end_points.append(i - 1)
            else:
                i += 1

        bead_count = len(start_points)
        indices_to_plot = range(bead_count)
        if bead_numbers:
            indices_to_plot = [i for i in indices_to_plot if i + 1 in bead_numbers]

        for col_idx, column in enumerate(df.columns[:3]):
            cumulative_index = 0
            for i in indices_to_plot:
                segment = df.iloc[start_points[i]:end_points[i] + 1]
                normalized_index = list(range(cumulative_index, cumulative_index + len(segment)))
                cumulative_index += len(segment)

                bead_data[col_idx].append({
                    "x": normalized_index,
                    "y": segment[column].values,
                    "tooltip": [
                        f"File: {file}<br>Bead: {i + 1}<br>Original Index: {idx}<br>Start Point: {start_points[i]}<br>End Point: {end_points[i]}<br>Value: {val}"
                        for idx, val in zip(segment.index, segment[column].values)
                    ],
                    "color": file_colors[file],
                    "bead_index": i
                })

                if i not in bead_segments_by_index[col_idx]:
                    bead_segments_by_index[col_idx][i] = []
                bead_segments_by_index[col_idx][i].append(segment[column].values)

    # Compute limits for each bead index
    for col_idx in range(3):
        for bead_index, segments in bead_segments_by_index[col_idx].items():
            min_len = min(map(len, segments))
            truncated_segments = [s[:min_len] for s in segments]
            stacked = np.vstack(truncated_segments)
            if method == "Standard Deviation":
                mean = np.mean(stacked, axis=0)
                std = np.std(stacked, axis=0)
                upper = pd.Series(mean + 2 * std).rolling(rolling_window, min_periods=1).mean()
                lower = pd.Series(mean - 2 * std).rolling(rolling_window, min_periods=1).mean()
            elif method == "Percentile":
                upper = pd.Series(np.percentile(stacked, 95, axis=0)).rolling(rolling_window, min_periods=1).mean()
                lower = pd.Series(np.percentile(stacked, 5, axis=0)).rolling(rolling_window, min_periods=1).mean()
            bead_limits[col_idx][bead_index] = (upper.values, lower.values)

    # Plotting
    fig_columns = [go.Figure() for _ in range(3)]

    for col_idx, fig in enumerate(fig_columns):
        column_name = sample_df.columns[col_idx]
        legend_shown = set()

        for data in bead_data[col_idx]:
            file_name = data["tooltip"][0].split("<br>")[0].split(": ")[1]
            show_legend = file_name not in legend_shown
            if show_legend:
                legend_shown.add(file_name)

            fig.add_trace(go.Scatter(
                x=data["x"],
                y=data["y"],
                mode='lines',
                name=file_name if show_legend else None,
                legendgroup=file_name,
                showlegend=show_legend,
                hoverinfo='text',
                text=data["tooltip"],
                line=dict(color=data["color"], width=0.5)
            ))

            bead_index = data["bead_index"]
            if bead_index in bead_limits[col_idx]:
                upper, lower = bead_limits[col_idx][bead_index]
                fig.add_trace(go.Scatter(
                    x=data["x"],
                    y=upper[:len(data["x"])],
                    mode='lines',
                    name=f"Upper Limit - Bead {bead_index + 1}",
                    line=dict(color='red', width=1, dash='dash'),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=data["x"],
                    y=lower[:len(data["x"])],
                    mode='lines',
                    name=f"Lower Limit - Bead {bead_index + 1}",
                    line=dict(color='blue', width=1, dash='dash'),
                    showlegend=False
                ))

        fig.update_layout(
            title=f"Visualization for {column_name}",
            xaxis_title="Normalized Index",
            yaxis_title="Signal Value",
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig)
