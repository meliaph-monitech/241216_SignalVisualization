import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample dataset for demonstration
# Replace this section with your actual data loading process
np.random.seed(42)
data = pd.DataFrame(
    np.random.rand(100, 5), 
    columns=["Column_A", "Column_B", "Column_C", "Column_D", "Column_E"]
)
data['Bead_Number'] = range(1, 101)  # Add Bead_Number column

# Title
st.title("Bead Segmentation and Visualization")

# User input for bead numbers
st.sidebar.header("Bead Segmentation and Visualization")
bead_numbers_input = st.sidebar.text_input(
    "Enter Bead Numbers to Visualize (default Bead No.1, blank for all):", 
    value="1"
)

# Parse input bead numbers
if bead_numbers_input.strip() == "":
    bead_numbers = data['Bead_Number']  # All bead numbers
else:
    try:
        bead_numbers = list(map(int, bead_numbers_input.split(",")))
    except ValueError:
        st.error("Invalid input. Please enter bead numbers as comma-separated values.")
        bead_numbers = []

# Filter data based on bead numbers
filtered_data = data[data['Bead_Number'].isin(bead_numbers)]

# Button to show all bead numbers in each column's line plot
if st.sidebar.button("Show All Bead Numbers for Each Column"):
    st.header("Line Plots of All Bead Numbers for Each Column")
    for column in data.columns:
        if column == "Bead_Number":
            continue  # Skip the Bead_Number column

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(data["Bead_Number"], data[column], label=f"{column}", marker="o")
        ax.set_title(f"Line Plot of {column}", fontsize=14)
        ax.set_xlabel("Bead Number")
        ax.set_ylabel(column)
        ax.grid(True)
        ax.legend()
        
        st.pyplot(fig)

# Show line plots for the selected bead numbers
if not filtered_data.empty:
    st.header("Line Plots for Selected Bead Numbers")
    for column in filtered_data.columns:
        if column == "Bead_Number":
            continue  # Skip the Bead_Number column

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(filtered_data["Bead_Number"], filtered_data[column], marker="o")
        ax.set_title(f"Line Plot of {column} (Selected Beads)", fontsize=14)
        ax.set_xlabel("Bead Number")
        ax.set_ylabel(column)
        ax.grid(True)
        
        st.pyplot(fig)
else:
    st.info("No bead numbers selected or invalid input.")
