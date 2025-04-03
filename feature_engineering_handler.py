import streamlit as st
import pandas as pd
import numpy as np
from history_manager import record_new_change

def display_feature_engineering(selected_dataset_name, df):
    """
    Provides an interface for manual feature engineering:
      - Users choose between transforming a single column or combining two columns.
      - Displays a real-time preview of the new feature.
      - The new feature is not immediately appended to the main dataset until the user confirms.
      - Multiple new features can be created persistently.
      - Also provides an option to delete a created feature.
    
    Returns:
        The updated DataFrame (with the new feature appended, if confirmed).
    """
    st.subheader("Feature Interaction")
    st.info("Create a new feature by combining two existing features.")

    # Ensure persistent storage for new features if not already created.
    if "fe_new_df" not in st.session_state:
        st.session_state["fe_new_df"] = df.copy()
    if "created_features" not in st.session_state:
        st.session_state["created_features"] = []

    # Filter columns to only numeric columns.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric features available for feature engineering.")
        return df
    
    # Arrange the UI in three columns.
    col1, col2, col3 = st.columns([3, 2, 3])
    
    with col1:
        cols = df.columns.tolist()
        feature1 = st.selectbox("Select First Feature", options=numeric_cols, key="fe_feature1")
    with col2:
        operation = st.selectbox("Select Operation", options=["Add", "Subtract", "Multiply", "Divide"], key="fe_operation")
    with col3:
        feature2 = st.selectbox("Select Second Feature", options=numeric_cols, key="fe_feature2")
    
    new_feature_name = st.text_input("New Feature Name", key="fe_new_feature")
    
    # New column layout below that
    col12, col22 = st.columns([3,3])

    # Button to preview and add the new feature to the temporary new dataset.
    with col12:
        if st.button("Apply Feature Engineering", key="fe_apply"):
            if not new_feature_name:
                st.error("Please provide a name for the new feature.")
            else:
                try:
                    temp_df = st.session_state["fe_new_df"].copy()
                    if operation == "Add":
                        temp_df[new_feature_name] = temp_df[feature1] + temp_df[feature2]
                    elif operation == "Subtract":
                        temp_df[new_feature_name] = temp_df[feature1] - temp_df[feature2]
                    elif operation == "Multiply":
                        temp_df[new_feature_name] = temp_df[feature1] * temp_df[feature2]
                    elif operation == "Divide":
                        # Avoid division by zero.
                        temp_df[new_feature_name] = temp_df[feature1] / temp_df[feature2].replace({0: pd.NA})
                    
                    st.session_state["fe_new_df"] = temp_df  # Update the persistent temporary dataset.
                    
                    # Add the new feature name to created_features list if not already present.
                    if new_feature_name not in st.session_state["created_features"]:
                        st.session_state["created_features"].append(new_feature_name)
                    st.success(f"Preview: New feature '{new_feature_name}' created.")
                
                except Exception as e:
                    st.error(f"Error creating new feature: {e}")
        
    # Button to confirm and update the main dataset with all new features.
    with col22:
        if st.button("Confirm & Update Features", key="fe_confirm"):
            try:
                # Update the main dataset with the temporary new features.
                updated_df = st.session_state["fe_new_df"]
                record_new_change(selected_dataset_name, updated_df, "Feature Engineering: New features confirmed.")
                st.success("Main dataset updated with new features!")
                # Clear the temporary new dataset and created features storage.
                st.session_state["fe_new_df"] = updated_df.copy()
                st.session_state["created_features"] = []
                st.rerun()
            except Exception as e:
                st.error(f"Error updating main dataset: {e}")

    # Display the preview of the new dataset with engineered features (only new columns added).
    created = st.session_state.get("created_features", [])
    if created:
        st.markdown("#### Preview of Engineered Features")
        st.dataframe(st.session_state["fe_new_df"][created].head(10))
    
    # Option to delete a created feature.
    if st.session_state["created_features"]:
        st.markdown("#### Delete a Created Feature")
        feature_to_delete = st.selectbox("Select Feature to Delete", options=st.session_state["created_features"], key="fe_delete_select")
        if st.button("Delete Feature", key="fe_delete"):
            try:
                temp_df = st.session_state["fe_new_df"].copy()
                if feature_to_delete in temp_df.columns:
                    temp_df.drop(columns=[feature_to_delete], inplace=True)
                    st.session_state["fe_new_df"] = temp_df.copy()
                    st.session_state["created_features"].remove(feature_to_delete)
                    record_new_change(selected_dataset_name, temp_df, f"Feature Engineering: Deleted feature '{feature_to_delete}'.")
                    st.success(f"Feature '{feature_to_delete}' deleted.")
                    st.rerun()
                else:
                    st.error("Feature not found in dataset.")
            except Exception as e:
                st.error(f"Error deleting feature: {e}")
    
    return st.session_state["fe_new_df"]