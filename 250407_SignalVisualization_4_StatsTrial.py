import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO

# Set page layout to wide
st.set_page_config(layout="wide")

# Streamlit Title
st.title("Laser Welding Signal Visualization")

# Step 1: Upload ZIP File
uploaded_zip = st.file_uploader("Upload a ZIP file containing CSV files:", type="zip")

if uploaded_zip:
    extract_dir = "extracted_csvs"
    if os.path.exists(extract_dir):
        for file_name in os.listdir(extract_dir):
            file_path = os.path.join(extract_dir, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                else:
                    os.rmdir(file_path)
            except Exception as e:
                st.error(f"Error cleaning up previous files: {e}")
        os.rmdir(extract_dir)

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]

    if csv_files:
        st.success(f"Extracted {len(csv_files)} CSV files.")

        st.write("### Filter Data")
        sample_df = pd.read_csv(os.path.join(extract_dir, csv_files[0]))
        column_names = sample_df.columns.tolist()

        filter_column = st.selectbox("Select the Filter Column to reduce data:", column_names)
        filter_threshold = st.number_input("Set Filter Threshold:", value=1.0)

        limit_method = st.selectbox("Choose Method for Limit Calculation:", ["Rolling Std Dev", "Percentile"])
        window_size = st.slider("Window Size for Rolling Calculation:", min_value=5, max_value=500, value=100)
        percentile_val = st.slider("Percentile Value (for Percentile method):", min_value=80, max_value=99, value=95)

        filter_triggered = st.button("Visualize")
        session_state = st.session_state

        if filter_triggered or "filtered_files" in session_state:
            st.write("Filtering data...")
            if "filtered_files" not in session_state or filter_triggered:
                session_state.filtered_files = {}
                for file in csv_files:
                    df = pd.read_csv(os.path.join(extract_dir, file))
                    if (df[filter_column] > filter_threshold).any():
                        session_state.filtered_files[file] = df

            st.success(f"Filtered down to {len(session_state.filtered_files)} files after applying the threshold.")

            st.write("### Select CSV Files")
            file_list = list(session_state.filtered_files.keys())
            selected_files = st.multiselect("Select CSV file(s) to visualize:", options=["All"] + file_list, default="All")

            if "All" in selected_files:
                selected_files = file_list

            if selected_files:
                st.write("### Bead Segmentation and Visualization")
                bead_numbers = st.text_input("Enter Bead Numbers to Visualize (default Bead No.1, blank for all):", value="1")
                bead_numbers = [int(b.strip()) for b in bead_numbers.split(',') if b.strip().isdigit()] if bead_numbers else None

                bead_data = {col_idx: [] for col_idx in range(3)}
                bead_dict = {col_idx: {} for col_idx in range(3)}
                file_colors = {file: f"rgb({(hash(file) % 256)},{(hash(file + 'g') % 256)},{(hash(file + 'b') % 256)})" for file in selected_files}

                for file in selected_files:
                    df = session_state.filtered_files[file]
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

                    indices_to_plot = range(len(start_points))
                    if bead_numbers:
                        indices_to_plot = [i for i in range(len(start_points)) if i + 1 in bead_numbers]

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
                                    f"File: {file}<br>Bead: {i + 1}<br>Original Index: {idx}<br>Value: {val}"
                                    for idx, val in zip(segment.index, segment[column].values)
                                ],
                                "color": file_colors[file]
                            })

                            if i + 1 not in bead_dict[col_idx]:
                                bead_dict[col_idx][i + 1] = []
                            bead_dict[col_idx][i + 1].append(segment[column].reset_index(drop=True))

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

                    for bead_no, segments in bead_dict[col_idx].items():
                        min_len = min(len(seg) for seg in segments)
                        aligned = [seg.iloc[:min_len] for seg in segments]
                        combined = pd.concat(aligned, axis=1)
                        mean_signal = combined.mean(axis=1)

                        if limit_method == "Rolling Std Dev":
                            std_signal = combined.std(axis=1)
                            upper = (mean_signal + std_signal).rolling(window=window_size, min_periods=1).mean()
                            lower = (mean_signal - std_signal).rolling(window=window_size, min_periods=1).mean()
                        else:
                            upper = combined.apply(lambda row: row.quantile(percentile_val / 100), axis=1).rolling(window=window_size, min_periods=1).mean()
                            lower = combined.apply(lambda row: row.quantile(1 - percentile_val / 100), axis=1).rolling(window=window_size, min_periods=1).mean()

                        x_vals = list(range(len(mean_signal)))
                        fig.add_trace(go.Scatter(x=x_vals, y=upper, mode='lines', name=f"Bead {bead_no} Upper", line=dict(color='red', dash='dash'), showlegend=False))
                        fig.add_trace(go.Scatter(x=x_vals, y=lower, mode='lines', name=f"Bead {bead_no} Lower", line=dict(color='blue', dash='dash'), showlegend=False))

                    fig.update_layout(
                        title=f"Visualization for {column_name}",
                        xaxis_title="Normalized Index",
                        yaxis_title="Signal Value",
                        height=600,
                        showlegend=True
                    )
                    st.plotly_chart(fig)
