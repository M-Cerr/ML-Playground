import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from agstyler import draw_grid

def display_train_test_split(selected_dataset_name, df, formatter=None):
    """
    Train/Test Splitting UI (no table hiding).
    
    1) If the user toggles the checkbox "Train/Test Splitting", they see a ratio slider
       unless they already have a train/test set in session state.
    2) Once the user sets a ratio and clicks "Apply Split", we store the train/test sets in
       st.session_state["train_dataset"] and st.session_state["test_dataset"] (not confirmed yet).
    3) We display smaller-sized tables for both train & test sets stacked.
    4) The user can either confirm the split (saving them in session for later usage) or restart.
    5) If the user unchecks the checkbox, we remove the train/test from session state, so the tool is effectively closed.
    6) The main table never disappears; these new tables appear below it.
    """

    if formatter is None:
        formatter = {}

    st.sidebar.title("Train/Test Splitting")

    # Check if user toggles the tool
    split_tool_on = st.sidebar.checkbox("Enable Train/Test Splitting")
    if not split_tool_on:
        # Instead of clearing the sets, we just do nothing, so that if user toggles back on,
        # we re-display the same sets if they exist.
        # If you want them to vanish on toggle off, you'd pop them here, but that causes the issue you mentioned.
        return

    # If user never used it before, init the confirm flag
    if "train_test_confirmed" not in st.session_state:
        st.session_state["train_test_confirmed"] = False

    has_split = ("train_dataset" in st.session_state and "test_dataset" in st.session_state)

    st.write("### Train/Test Splitting Section")

    # If no sets exist, show ratio slider
    if not has_split:
        if st.session_state["train_test_confirmed"]:
            # If confirmed was previously set but sets are missing, we reset
            st.session_state["train_test_confirmed"] = False

        split_ratio = st.slider(
            "Select Train/Test Split Ratio",
            min_value=0.1, max_value=0.9,
            value=0.8, step=0.05,
            help="Proportion of data for training."
        )

        if st.button("Apply Split"):
            try:
                train_df, test_df = train_test_split(df, test_size=1 - split_ratio, random_state=42)
                train_df = train_df.reset_index().rename(columns={"index": "Index"})
                test_df = test_df.reset_index().rename(columns={"index": "Index"})
                st.session_state["train_dataset"] = train_df
                st.session_state["test_dataset"] = test_df
                st.success(f"Split Created! (Train: {len(train_df)}, Test: {len(test_df)})")
            except Exception as e:
                st.error(f"Error splitting: {e}")
    else:
        # Sets exist in session, so we display them. If user never confirmed, the main table remains visible too.
        train_df = st.session_state["train_dataset"]
        test_df = st.session_state["test_dataset"]

        st.write("Train/Test sets already exist in session. View below or confirm them.")

    # If sets exist, display them
    if "train_dataset" in st.session_state and "test_dataset" in st.session_state:
        train_df = st.session_state["train_dataset"]
        test_df = st.session_state["test_dataset"]

        st.write("#### Training Dataset")
        draw_grid(
            train_df,
            formatter=formatter,
            fit_columns=False,   # smaller, so they stand out from the main table
            theme="streamlit",
            max_height=300,
            key="tt_train_grid"
        )

        st.write("#### Testing Dataset")
        draw_grid(
            test_df,
            formatter=formatter,
            fit_columns=False,
            theme="streamlit",
            max_height=300,
            key="tt_test_grid"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Training Set",
                data=train_df.to_csv(index=False),
                file_name=f"{selected_dataset_name}_train.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                label="📥 Download Testing Set",
                data=test_df.to_csv(index=False),
                file_name=f"{selected_dataset_name}_test.csv",
                mime="text/csv"
            )

        # Confirmation button if not confirmed
        if not st.session_state["train_test_confirmed"]:
            if st.button("Confirm Split"):
                st.session_state["train_test_confirmed"] = True
                st.success("Train/Test Split Confirmed. Sets retained for future usage.")
        else:
            st.info("Train/Test Split is already confirmed. You can use these sets in future steps.")

        # A "Restart" button to remove sets from session
        if st.button("Restart Splitting"):
            st.session_state.pop("train_dataset", None)
            st.session_state.pop("test_dataset", None)
            st.session_state["train_test_confirmed"] = False
            st.info("Train/Test split cleared. You can re-apply a new ratio. Please close and re-open tool.")

    # End: we do not hide or modify the main table; user sees everything
    return
