import streamlit as st

st.title("ML PLayground")
st.write("Welcome to the ML Playground Tool :)")

st.sidebar.title("Preprocessing Options")
st.sidebar.checkbox("Missing Value Replacement")
st.sidebar.checkbox("Scaling")
