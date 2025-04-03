import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from history_manager import DatasetHistory, record_new_change


def display_scaling_options(selected_dataset_name, df):
    """
    Handles UI for data scaling and normalization.
    Allows users to apply Min-Max Scaling or Z-Score Standardization to numeric columns.

    Args:
        - selected_dataset_name: Name of the selected dataset.
        - df: The dataset (DataFrame).

    Returns:
        - updated_df: The modified dataset after scaling.
    """

    if st.sidebar.checkbox("Scaling & Normalization"):
        st.write("### Scale or Normalize Numeric Data")

        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_columns:
            st.warning("No numeric columns found in this dataset.")
            return df

        # Initialize session state for temp dataset storage
        if "temp_scaled_dataset" not in st.session_state:
            st.session_state["temp_scaled_dataset"] = df.copy()

        # Select columns to scale
        selected_columns = st.multiselect(
            "Select Columns to Scale",
            options=numeric_columns,
            default=numeric_columns,  # Default to all numeric columns
            help="Choose numeric columns to scale."
        )

        # Select scaling method
        scaling_method = st.selectbox(
            "Select Scaling Method",
            ["Min-Max Scaling (0-1)", "Z-Score Standardization"],
            help="Min-Max Scaling scales values between 0 and 1. Z-Score Standardization makes mean = 0, std dev = 1."
        )

        # Apply Scaling
        if st.button("Apply Scaling"):
            try:
                temp_df = st.session_state["temp_scaled_dataset"].copy()

                if scaling_method == "Min-Max Scaling (0-1)":
                    scaler = MinMaxScaler()
                else:  # Z-Score Standardization
                    scaler = StandardScaler()

                temp_df[selected_columns] = scaler.fit_transform(temp_df[selected_columns])
                st.session_state["temp_scaled_dataset"] = temp_df  # Store updated dataset

                st.success("Scaling applied! You can continue scaling other columns or confirm changes.")
            except Exception as e:
                st.error(f"Error during scaling: {e}")

        # Show Before & After Comparison
        if st.session_state["temp_scaled_dataset"] is not None:
            st.write("### Original vs. Scaled Dataset")

            col1, col2 = st.columns(2)

            with col1:
                st.write("#### 🔹 Original Dataset (Before Scaling)")
                st.dataframe(df)

            with col2:
                st.write("#### 🔹 Scaled Dataset (After Scaling)")
                st.dataframe(st.session_state["temp_scaled_dataset"])

            # Only Update Main Dataset When User Confirms
            if st.button("Confirm & Update Dataset", key="scalingConfirm"):
                try:
                    # Update main dataset
                    st.session_state["datasets"][selected_dataset_name] = st.session_state["temp_scaled_dataset"]
                    record_new_change(
                        selected_dataset_name,
                        st.session_state["temp_scaled_dataset"],
                        "Scaling tool changes confirmed."
                    )
                    st.success("Main dataset updated successfully!")

                    # Clear temporary dataset & refresh UI
                    del st.session_state["temp_scaled_dataset"]
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating main dataset: {e}")

    return df  # Return unchanged DataFrame if feature is not selected
