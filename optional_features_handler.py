import streamlit as st
import pandas as pd
import numpy as np
from mutual_information_handler import display_mutual_information
from feature_engineering_handler import display_feature_engineering
from ui_explanations import display_section_header_with_help


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
            st.subheader("Mutual Information Analysis")
            with st.expander(f"What does this tool do?", expanded=False):
              st.write("**Mutual Information (MI):** MI quantifies the amount of shared information between a feature and the target variable. "
              "A higher MI score means that the feature provides more predictive power about the target. "
              "Use these scores to identify the most relevant features for your model.")
            
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