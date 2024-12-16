# Fifth

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
    # Extract ZIP file contents
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        # Create a temporary directory to extract files
        extract_dir = "extracted_csvs"
        os.makedirs(extract_dir, exist_ok=True)
        zip_ref.extractall(extract_dir)

    # List all CSV files extracted
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]

    if csv_files:
        st.success(f"Extracted {len(csv_files)} CSV files.")

        # Display the list of CSV files
        selected_files = st.multiselect(
            "Select CSV file(s) to visualize:", csv_files
        )

        if selected_files:
            # Load the selected files
            dfs = {file: pd.read_csv(os.path.join(extract_dir, file)) for file in selected_files}

            # Step 2: Filter Column Selection
            st.write("### Filter Column Selection")
            common_columns = dfs[selected_files[0]].columns.tolist()
            filter_column = st.selectbox("Select the Filter Column:", common_columns)

            # Step 3: Input Filter Threshold
            filter_threshold = st.number_input("Set Filter Threshold:", value=1.0)

            # Step 4: Bead Segmentation and Visualization
            st.write("### Bead Segmentation and Visualization")
            bead_data = {}

            for file, df in dfs.items():
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

                bead_data[file] = {
                    "start_points": start_points,
                    "end_points": end_points
                }

            # Visualization Controls
            bead_numbers = st.text_input("Enter Bead Numbers to Visualize (comma-separated, or leave blank for all):")
            bead_numbers = [int(b.strip()) for b in bead_numbers.split(',') if b.strip().isdigit()] if bead_numbers else None

            # Normalize Indices and Prepare Data for Plotting
            normalized_data = {col_idx: [] for col_idx in range(3)}

            for file, df in dfs.items():
                bead_indices = bead_data[file]
                start_points = bead_indices["start_points"]
                end_points = bead_indices["end_points"]

                if bead_numbers:
                    indices_to_plot = [i for i in range(len(start_points)) if i + 1 in bead_numbers]
                else:
                    indices_to_plot = range(len(start_points))

                for col_idx, column in enumerate(df.columns[:3]):
                    cumulative_index = 0
                    for i in indices_to_plot:
                        segment = df.iloc[start_points[i]:end_points[i] + 1]
                        normalized_index = list(range(cumulative_index, cumulative_index + len(segment)))
                        cumulative_index += len(segment)
                        normalized_data[col_idx].append({
                            "x": normalized_index,
                            "y": segment[column].values,
                            "tooltip": [f"File: {file}<br>Bead: {i + 1}" for _ in range(len(segment))]
                        })

            # Plotting with Plotly
            fig_columns = [go.Figure() for _ in range(3)]

            for col_idx, fig in enumerate(fig_columns):
                for data in normalized_data[col_idx]:
                    fig.add_trace(go.Scatter(
                        x=data["x"],
                        y=data["y"],
                        mode='lines',
                        hoverinfo='text',
                        text=data["tooltip"]
                    ))
                fig.update_layout(
                    title=f"Visualization for Column {col_idx + 1}",
                    xaxis_title="Normalized Index",
                    yaxis_title="Signal Value",
                    height=600
                )
                st.plotly_chart(fig)

            st.write("Visualization Complete!")
