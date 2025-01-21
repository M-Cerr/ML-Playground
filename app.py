import streamlit as st
from data_loader import load_sample_datasets

# Initialize session state for datasets
if 'datasets' not in st.session_state:
    st.session_state['datasets'] = load_sample_datasets()

st.title("ML PLayground")
st.write("Welcome to the ML Playground Tool :)")

# Dropdown to select dataset
selected_dataset = st.selectbox("Choose a dataset to view", options=list(st.session_state['datasets'].keys()))

# Display the selected dataset
if selected_dataset:
    st.write(f"Displaying {selected_dataset}")
    st.dataframe(st.session_state['datasets'][selected_dataset])

st.sidebar.title("Preprocessing Options")
st.sidebar.checkbox("Missing Value Replacement")
st.sidebar.checkbox("Scaling")
