import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import altair as alt

def display_mutual_information(selected_dataset_name, df):
    """
    Displays mutual information scores between each numeric feature and a user-selected target feature.
    The results are shown as a bar chart by default with an option to switch to a sorted table.
    
    Returns:
        mi_df: A DataFrame of mutual information scores.
    """
    #st.subheader("Analysis")
    
    # Determine numeric columns in df.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    #To filter out primary key
    primary_key = st.session_state.get(f"{selected_dataset_name}_primary_key")
    #To filter out any encoded features
    original_cols = st.session_state.get("original_columns", {}).get(selected_dataset_name, [])
    encoded_features = list(set(df.columns) - set(original_cols))

    if not numeric_cols:
        st.warning("No numeric features available for mutual information analysis.")
        return None

    # Check for missing values in numeric columns.
    cols_with_missing = [col for col in numeric_cols if df[col].isna().any()]
    if cols_with_missing:
        st.warning(f"Numeric Features: {', '.join(cols_with_missing)} contain missing values. Please ensure none of the numeric features contain missing values before using this tool.")
        return None
    
    target_choices = [col for col in numeric_cols if col != primary_key and col not in encoded_features]
    # Ask the user to select a target feature.
    target_feature = st.selectbox("Select Target Feature", options=target_choices, key="mi_target")
    if not target_feature:
        st.info("Please select a target feature for MI analysis.")
        return None


    # Use mutual_info_regression to compute MI scores for features (excluding the target).
    features = [col for col in numeric_cols if col != target_feature and col != primary_key]
    if not features:
        st.info("Not enough numeric features for MI analysis (need at least one other than target).")
        return None

    X = df[features]
    y = df[target_feature]
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({
        "Feature": features,
        "MI Score": mi_scores
    }).sort_values(by="MI Score", ascending=False)

    # Choose display mode: Bar chart (default) or Table.
    display_mode = st.radio("Display Mutual Information as", ["Bar Chart", "Table"], key="mi_display")
    if display_mode == "Bar Chart":
        chart = alt.Chart(mi_df).mark_bar().encode(
            x=alt.X("MI Score:Q", title="Mutual Information Score"),
            y=alt.Y("Feature:N", sort="-x", title="Feature"),
            tooltip=["Feature", "MI Score"]
        ).properties(
            width=300,
            height=200,
            title="Mutual Information Scores"
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.dataframe(mi_df.reset_index(drop=True))
    
    # Optionally, provide a download button for the MI scores.
    csv = mi_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download MI Scores as CSV",
        data=csv,
        file_name=f"{selected_dataset_name}_MI_scores.csv",
        mime="text/csv"
    )
    
    return mi_df