import streamlit as st
import pandas as pd
from preprocessing import apply_categorical_encoding
from history_manager import DatasetHistory, record_new_change

def display_categorical_encoding(selected_dataset_name, df):
    """
    Handles UI for categorical data encoding.
    Allows users to select encoding methods and apply them interactively.

    Args:
        - selected_dataset_name: Name of the selected dataset.
        - df: The dataset (DataFrame).

    Returns:
        - updated_df: The modified dataset after encoding.
    """

    if st.sidebar.checkbox("Categorical Data Encoding"):
        st.write("### Encode Categorical Data")

        # Retrieve categorical columns
        categorical_columns = st.session_state['categorical_columns'].get(selected_dataset_name, [])

        # Ensure valid columns exist in the dataset
        available_columns = [col for col in categorical_columns if col in df.columns]

        # Initialize session states for encoding
        if "selected_columns" not in st.session_state:
            st.session_state["selected_columns"] = []

        if "temp_encoded_dataset" not in st.session_state:
            st.session_state["temp_encoded_dataset"] = df.copy()

        selected_columns = st.multiselect(
            "Select Columns to Encode",
            options=available_columns,
            default=st.session_state["selected_columns"],  
            help="Choose categorical columns to transform into numeric values."
        )

        # Persist selection state
        if selected_columns != st.session_state["selected_columns"]:
            st.session_state["selected_columns"] = selected_columns

        encoding_method = st.selectbox(
            "Select Encoding Method",
            ["One-Hot Encoding", "Label Encoding", "Ordinal Encoding", "Count Encoding"],
            help="Select how to encode categorical variables."
        )

        # Additional Options for Each Encoding Type
        encoding_params = {}

        if encoding_method == "One-Hot Encoding":
            encoding_params["drop_option"] = st.radio(
                "Select Column Dropping Behavior",
                ["Drop first column in all features", "Drop first column if binary feature", "Keep all columns"],
                index=2  # Default: Keep all columns
            )

        elif encoding_method == "Label Encoding":
            encoding_params["handle_unknown"] = st.radio(
                "Handling Unknown Categories",
                ["Error", "Assign -1", "Ignore"],
                index=1  # Default: Assign -1
            )

        elif encoding_method == "Ordinal Encoding":
            st.write("Define Custom Order for Each Selected Column:")
            encoding_params["custom_order"] = {}
            for col in selected_columns:
                unique_values = df[col].dropna().unique().tolist()
                encoding_params["custom_order"][col] = st.multiselect(
                    f"Define Order for `{col}`",
                    options=unique_values,
                    default=unique_values,  # Default to detected unique values
                    help="Rearrange the order as needed."
                )

        elif encoding_method == "Count Encoding":
            encoding_params["apply_log"] = st.checkbox(
                "Apply Log Transformation",
                value=False,
                help="Enable to normalize large count variations."
            )

        # Apply Encoding Without Resetting Previous Changes
        if st.button("Apply Encoding"):
            try:
                temp_df = apply_categorical_encoding(
                    #st.session_state["temp_encoded_dataset"],  # Use cumulative dataset
                    df,
                    selected_columns,
                    encoding_method,
                    encoding_params
                )
                st.session_state["temp_encoded_dataset"] = temp_df  # Store updated dataset

                st.success("Encoding applied! You can continue encoding other columns or confirm changes.")
            except Exception as e:
                st.error(f"Error while encoding categorical data: {e}")

        # Show Before & After Side-by-Side Comparison
        if st.session_state["temp_encoded_dataset"] is not None:
            st.write("### Original vs. Encoded Dataset")

            col1, col2 = st.columns(2)

            with col1:
                st.write("#### 🔹 Original Dataset (Before Encoding)")
                st.dataframe(df)

            with col2:
                st.write("#### 🔹 Encoded Dataset (After Encoding)")
                st.dataframe(st.session_state["temp_encoded_dataset"])

            # Only Update the Main Dataset When User Confirms
            if st.button("Confirm & Update Dataset"):
                try:
                    # Ensure only existing columns remain in session state
                    dataset_columns = st.session_state["temp_encoded_dataset"].columns.tolist()
                    st.session_state['categorical_columns'][selected_dataset_name] = [
                        col for col in st.session_state['categorical_columns'].get(selected_dataset_name, []) if col in dataset_columns
                    ]

                    # Reset selected columns to prevent reference errors
                    st.session_state["selected_columns"] = []

                    # Update main dataset
                    st.session_state['datasets'][selected_dataset_name] = st.session_state["temp_encoded_dataset"]
                    new_df = st.session_state["temp_encoded_dataset"]
                    record_new_change(
                        selected_dataset_name,
                        new_df,
                        "Categorical encoding change confirmed."
                    )
                    st.success("Main dataset updated successfully!")

                    # Clear temporary dataset & refresh UI
                    del st.session_state["temp_encoded_dataset"]
                    st.rerun()

                except Exception as e:
                    st.error(f"Error updating main dataset: {e}")

    return df  # Return unchanged DataFrame if feature is not selected
