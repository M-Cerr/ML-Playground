import streamlit as st
import pandas as pd
import numpy as np
from mutual_information_handler import display_mutual_information
from feature_engineering_handler import display_feature_engineering

def display_optional_features(selected_dataset_name, df):
    """
    Displays optional feature engineering tools (Mutual Information and Feature Engineering)
    in the main area using a multi-column layout.
    
    Parameters:
      - selected_dataset_name: The normalized key for the dataset.
      - df: The current DataFrame.
      
    Returns:
      The (potentially) modified DataFrame after any optional modifications.
    """
    st.sidebar.title("Optional Features")
    
    # Sidebar checkboxes to enable each tool.
    enable_mi = st.sidebar.checkbox("Enable Mutual Information", key="enable_mi")
    enable_fe = st.sidebar.checkbox("Enable Feature Interaction", key="enable_fe")
    
    # Create two columns.
    col1, col2 = st.columns(2)
    
    # In Column 1: Mutual Information Tool
    if enable_mi:
        with col1:
            st.subheader("Mutual Information")
            st.write("View MI scores between features and the target.")
            mi_results = display_mutual_information(selected_dataset_name, df)
            # Optionally, store mi_results or display additional options.
            st.markdown("---")
    
    # In Column 2: Feature Engineering Tool
    if enable_fe:
        with col2:
            st.subheader("Feature Engineering")
            st.write("Create new features by combining existing ones.")
            df = display_feature_engineering(selected_dataset_name, df)
            st.markdown("---")
    
    
    
    return df