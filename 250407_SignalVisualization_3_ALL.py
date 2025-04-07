import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
        bead_numbers = list(map(int, bead_numbers_input
