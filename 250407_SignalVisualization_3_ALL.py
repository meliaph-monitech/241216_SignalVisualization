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

        # Step 2: Filter Column Selection and Threshold Input
        st.write("### Filter Data")
        sample_df = pd.read_csv(os.path.join(extract_dir, csv_files[0]))
        column_names = sample_df.columns.tolist()

        # Select filter column
        filter_column = st.selectbox("Select the Filter Column to reduce data:", column_names)

        # Set filter threshold
        filter_threshold = st.number_input("Set Filter Threshold:", value=1.0)

        # Button to trigger filtering
        filter_triggered = st.button("Visualize")
        session_state = st.session_state

        if filter_triggered or "filtered_files" in session_state:
            # Filter data across all files to reduce processing
            st.write("Filtering data...")
            if "filtered_files" not in session_state or filter_triggered:
                session_state.filtered_files = {}
                for file in csv_files:
                    df = pd.read_csv(os.path.join(extract_dir, file))
                    # Apply filtering based on threshold
                    if (df[filter_column] > filter_threshold).any():
                        session_state.filtered_files[file] = df

            st.success(f"Filtered down to {len(session_state.filtered_files)} files after applying the threshold.")

            # Step 3: File Selection Dropdown with 'All' Option
            st.write("### Select CSV Files")
            file_list = list(session_state.filtered_files.keys())
            selected_files = st.multiselect("Select CSV file(s) to visualize:", options=["All"] + file_list, default="All")

            if "All" in selected_files:
                selected_files = file_list

            if selected_files:
                # Step 4: Bead Segmentation Preparation
                st.write("### Bead Segmentation and Visualization")
                bead_numbers = st.text_input("Enter Bead Numbers to Visualize (default Bead No.1, blank for all):", value="1")
                bead_numbers = [int(b.strip()) for b in bead_numbers.split(',') if b.strip().isdigit()] if bead_numbers else None

                # Normalize Indices and Prepare Data for Plotting
                bead_data = {col_idx: [] for col_idx in range(3)}
                file_colors = {file: f"rgb({(hash(file) % 256)},{(hash(file + 'g') % 256)},{(hash(file + 'b') % 256)})" for file in selected_files}

                for file in selected_files:
                    df = session_state.filtered_files[file]
                    filter_values = df[filter_column].to_numpy()

                    # Bead segmentation logic
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

                    # Prepare normalized data
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
                                    f"File: {file}<br>Bead: {i + 1}<br>Original Index: {idx}<br>Start Point: {start_points[i]}<br>End Point: {end_points[i]}<br>Value: {val}" \
                                    for idx, val in zip(segment.index, segment[column].values)
                                ],
                                "color": file_colors[file]
                            })
                
                # Plotting with Plotly
                fig_columns = [go.Figure() for _ in range(3)]
                
                for col_idx, fig in enumerate(fig_columns):
                    column_name = sample_df.columns[col_idx]  # Get actual column name
                    for data in bead_data[col_idx]:
                        fig.add_trace(go.Scatter(
                            x=data["x"],
                            y=data["y"],
                            mode='lines',
                            name=f"{data['tooltip'][0].split('<br>')[0]} - Bead {data['tooltip'][0].split('<br>')[1].split()[-1]}",  # Use File and Bead info for legend
                            hoverinfo='text',
                            text=data["tooltip"],
                            line=dict(color=data["color"], width=0.5),
                            showlegend=True
                        ))
                    fig.update_layout(
                        title=f"Visualization for {column_name}",
                        xaxis_title="Normalized Index",
                        yaxis_title="Signal Value",
                        height=600,
                        showlegend=True  # Show the legend so toggling works
                    )
                    st.plotly_chart(fig)
