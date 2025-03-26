from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, ColumnsAutoSizeMode
import streamlit as st
from agstyler import draw_grid, highlight, PRECISION_TWO, PINLEFT
from data_loader import load_sample_datasets
from dataset_analysis import analyze_dataset
from dataset_analysis import highlight_issues_aggrid
from preprocessing import handle_missing_values
from preprocessing import apply_categorical_encoding  # Import encoding function
import pandas as pd
import numpy as np

# Import the dataset selection & analysis UI from ui_main.py
from ui_main import display_dataset_selection_and_analysis
# Import Missing Value Replacement UI
from missing_value_handler import display_missing_value_replacement
# Import Categorical Encoding UI
from categorical_encoding_handler import display_categorical_encoding
# Import Scaling & Normalization UI
from scaling_handler import display_scaling_options
# Import Train/Test Splitting UI
from train_test_split_handler import display_train_test_split
# Import Outlier Detection UI
from outlier_detection_handler import display_outlier_detection
# Import History Modal UI
from ui_history import display_history_ui
# Import Optional (MI, Feature Interactions) UI
from optional_features_handler import display_optional_features


# Streamlit app
def main():
    st.set_page_config(  
    layout="wide" # <--- EXTREMELY IMPORTANT
    )

    # Initialize session state for datasets
    if 'datasets' not in st.session_state:
        st.session_state['datasets'] = load_sample_datasets()
    if 'categorical_columns' not in st.session_state:
        st.session_state['categorical_columns'] = {}
    if "histories" not in st.session_state:
        st.session_state["histories"] = {}  # mapping from dataset_name -> DatasetHistory

    # Step 1–3: Dataset selection, categorical column marking, and AgGrid table
    selected_dataset_name, display_title, df, updated_df, my_format = display_dataset_selection_and_analysis()
    if selected_dataset_name:
        # Only display history UI if the user has confirmed categorical columns.
        if st.session_state.get(f"{selected_dataset_name}_done", False):
            display_history_ui(selected_dataset_name)

    # Step 4: Missing Value Replacement (only runs if a dataset is selected)
    if selected_dataset_name and updated_df is not None:
        # Re-analyze dataset for issues before passing to missing value handler
        issues = analyze_dataset(updated_df, st.session_state['categorical_columns'].get(selected_dataset_name, []))
        updated_df = display_missing_value_replacement(selected_dataset_name, updated_df, issues)

        # Step 5: Categorical Data Encoding
        updated_df = display_categorical_encoding(selected_dataset_name, updated_df)

        # Step 6: Scaling & Normalization
        updated_df = display_scaling_options(selected_dataset_name, updated_df)

        # Step 7: Outlier Detection
        updated_df = display_outlier_detection(selected_dataset_name, updated_df)

        # Step 8: Mutual Information & Feature Interaction Display
        updated_df = display_optional_features(selected_dataset_name, updated_df)

        # Step 9: Train/Test Splitting
        display_train_test_split(selected_dataset_name, updated_df, my_format)

        # Update session state with the modified dataset
        st.session_state['datasets'][selected_dataset_name] = updated_df


if __name__ == "__main__":
    main()