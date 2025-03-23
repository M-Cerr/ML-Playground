import streamlit as st
import pandas as pd
import numpy as np
from history_manager import DatasetHistory, record_new_change


def detect_outliers_zscore(df, col, threshold=3.0):
    """
    Returns a set of row indices where |zscore| > threshold in the given column.
    """
    values = df[col].to_numpy()
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return set()
    zscores = (values - mean) / std
    return {i for i, z in enumerate(zscores) if abs(z) > threshold}

def detect_outliers_iqr(df, col, factor=1.5):
    """
    Returns a set of row indices where values are outside [Q1 - factor*IQR, Q3 + factor*IQR].
    """
    values = df[col].to_numpy()
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return {i for i, val in enumerate(values) if val < lower or val > upper}

def display_outlier_detection(selected_dataset_name, df):
    """
    Three-phase Outlier Detection tool with immediate table update.
    We store the updated df back into st.session_state['datasets'][selected_dataset_name] after each step,
    so the main table can reflect changes immediately upon st.rerun().

    Phase 0 (inactive): user must check "Enable Outlier Detection" in the sidebar or else we skip everything.
    Phase 1: user picks detection method & numeric columns, clicks "Compute Outliers".
        We compute sets => store them in st.session_state["outlier_indices"] => st.session_state["outlier_phase"] = 2 => st.rerun().
    Phase 2: user picks "Remove rows" or "Label in new column" => we update df => st.session_state["outlier_phase"] = 3 => st.rerun().
    Phase 3: final message => user sees updated table. They can uncheck tool or repeat.

    No outlier highlighting is done in the main table – we only remove or label them.
    """

    st.sidebar.title("Outlier Detection")
    enable_outlier = st.sidebar.checkbox("Enable Outlier Detection")
    if not enable_outlier:
        # If the user disables the tool, reset ephemeral data
        st.session_state.pop("outlier_phase", None)
        st.session_state.pop("outlier_indices", None)
        st.session_state.pop("outlier_chosen_cols", None)
        st.session_state.pop("outlier_method", None)
        return df  # no changes

    # if we haven't stored a "phase" in session, default to 1
    if "outlier_phase" not in st.session_state:
        st.session_state["outlier_phase"] = 1
    if "outlier_indices" not in st.session_state:
        st.session_state["outlier_indices"] = {}

    phase = st.session_state["outlier_phase"]

    # ---------------------------------------------
    # PHASE 1: DETECTION SETUP & "COMPUTE OUTLIERS"
    # ---------------------------------------------
    if phase == 1:
        st.write("### Phase 1: Detect Outliers")
        method = st.radio("Detection Method", ["Z-score", "IQR"])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns found for outlier detection.")
            return df

        chosen_cols = st.multiselect(
            "Select numeric columns:",
            numeric_cols,
            default=numeric_cols
        )

        if st.button("Compute Outliers"):
            # compute outlier sets
            st.session_state["outlier_indices"].clear()
            for col in chosen_cols:
                if method == "Z-score":
                    outlier_set = detect_outliers_zscore(df, col)
                else:
                    outlier_set = detect_outliers_iqr(df, col)
                st.session_state["outlier_indices"][col] = outlier_set

            st.session_state["outlier_method"] = method
            st.session_state["outlier_chosen_cols"] = chosen_cols
            st.session_state["outlier_phase"] = 2

            st.success(f"Outliers computed for columns={chosen_cols}. Proceed to Phase 2.")
            # store updated df in session, though we haven't changed df yet
            st.session_state['datasets'][selected_dataset_name] = df
            st.rerun()

        return df

    # ---------------------------------------------
    # PHASE 2: USER DECIDES "REMOVE" OR "LABEL"
    # ---------------------------------------------
    if phase == 2:
        st.write("### Phase 2: Handle Outliers")
        chosen_cols = st.session_state.get("outlier_chosen_cols", [])
        if not chosen_cols:
            st.warning("No columns chosen from Phase 1. Reverting to Phase 1.")
            st.session_state["outlier_phase"] = 1
            st.rerun()
            return df

        action = st.radio("Action on Outliers?", ["Remove rows", "Label in new column"])
        if st.button("Apply Outlier Action"):
            df_before = len(df)
            for col, outliers_set in st.session_state["outlier_indices"].items():
                if col not in chosen_cols:
                    continue
                if action == "Remove rows":
                    outliers_sorted = sorted(list(outliers_set), reverse=True)
                    for idx in outliers_sorted:
                        if idx < len(df):
                            df.drop(df.index[idx], inplace=True)
                else:
                    # label them in a hidden boolean col
                    label_col = f"is_outlier_{col}"
                    if label_col not in df.columns:
                        df[label_col] = False
                    for row_idx in outliers_set:
                        if row_idx < len(df):
                            df.iat[row_idx, df.columns.get_loc(label_col)] = True

            df_after = len(df)
            if action == "Remove rows":
                st.success(f"Removed {df_before - df_after} outlier rows.")
            else:
                st.info("Outlier labeling complete. The new 'is_outlier_{col}' columns are present in your dataset.")

            # finalize
            st.session_state['datasets'][selected_dataset_name] = df
            record_new_change(
                selected_dataset_name,
                df,
                "Outlier Tool used."
            )
            # clear ephemeral sets
            st.session_state["outlier_indices"].clear()
            # move to phase 3
            st.session_state["outlier_phase"] = 3
            st.rerun()

        return df

    # ---------------------------------------------
    # PHASE 3: DONE
    # ---------------------------------------------
    if phase == 3:
        st.write("### Phase 3: Outlier Detection Complete")
        st.info("Table updated. If you want to run outlier detection again, uncheck the box or change columns.")
        return df
