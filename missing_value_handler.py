import streamlit as st
import pandas as pd
from preprocessing import handle_missing_values
from agstyler import draw_grid

def display_missing_value_replacement(selected_dataset_name, df, issues):
    """
    Handles UI for missing value replacement.
    Displays missing value columns in an editable AgGrid table for user input and updates dataset accordingly.

    Args:
        - selected_dataset_name: Name of the selected dataset.
        - df: The dataset (DataFrame).
        - issues: The detected dataset issues (missing values & type mismatches).

    Returns:
        - updated_df: The modified dataset after missing value replacement.
    """

    # Display sidebar summary of missing values and type mismatches
    st.sidebar.header("Dataset Analysis Summary")

    # Missing values summary with expanders
    with st.sidebar.expander("⚠️ Missing Values Found" if issues["missing_values"] else "✅ No missing values detected"):
        if issues["missing_values"]:
            for col, rows in issues["missing_values"].items():
                st.write(f"`{col}`: Rows {rows}")

    # Type mismatches summary with expanders
    with st.sidebar.expander("⚠️ Type Mismatches Found" if issues["type_mismatches"] else "✅ No type mismatches detected"):
        if issues["type_mismatches"]:
            for col, rows in issues["type_mismatches"].items():
                st.write(f"`{col}`: Rows {rows}")

    # Categorical columns summary
    st.sidebar.write("### Categorical Columns")
    if st.session_state['categorical_columns'][selected_dataset_name]:
        for col in st.session_state['categorical_columns'][selected_dataset_name]:
            st.sidebar.write(f"- `{col}`")
    else:
        st.sidebar.write("None selected.")

    st.sidebar.title("Preprocessing Options")

    # Missing Values Section
    if st.sidebar.checkbox("Missing Value Replacement"):
        st.write("### Handle Missing Values")

        missing_columns = list(issues["missing_values"].keys())

        if missing_columns:
            # Create a DataFrame for Missing Value Handling
            missing_df = pd.DataFrame({
                "Column": missing_columns,
                "Column Type": ["Categorical" if col in st.session_state['categorical_columns'][selected_dataset_name] else "Numeric" for col in missing_columns],
                "Method": ["Mean" if col not in st.session_state['categorical_columns'][selected_dataset_name] else "Mode" for col in missing_columns],
                "Custom Value": ["" for _ in missing_columns],
            })

            # Define Formatter for Grid
            formatter = {}
            for col in missing_df.columns:
                col_props = {"width": 200}  # Default width for clarity
                if col == "Custom Value":
                    col_props["editable"] = True  # Allow user input
                elif col == "Method":
                    col_props["editable"] = True
                    col_props["cellEditor"] = "agSelectCellEditor"
                    col_props["cellEditorParams"] = {"values": ["Mean", "Median", "Mode", "Custom Value"]}
                formatter[col] = (col, col_props)

            # Display Editable Grid Using `draw_grid()`
            grid_response = draw_grid(
                missing_df,
                formatter=formatter,
                fit_columns=True,
                theme="streamlit",
                max_height=700,
            )

            # Get Updated Data from Grid
            updated_missing_df = pd.DataFrame(grid_response["data"])

            # Apply Fixes When User Clicks the Button
            if st.button("Apply Missing Value Fixes"):
                try:
                    for _, row in updated_missing_df.iterrows():
                        col = row["Column"]
                        method = row["Method"]
                        custom_value = row["Custom Value"] if row["Method"] == "Custom Value" else None
                        is_categorical = col in st.session_state['categorical_columns'][selected_dataset_name]

                        st.session_state['datasets'][selected_dataset_name] = handle_missing_values(
                            st.session_state['datasets'][selected_dataset_name],
                            col,
                            method,
                            custom_value,
                            is_categorical
                        )

                    st.success("Missing values handled successfully!")
                    st.rerun()  # Refresh the page to update the grid
                except Exception as e:
                    st.error(f"Error while handling missing values: {e}")

        else:
            st.write("✅ No missing values detected!")

    return df  # Return unchanged DataFrame if feature is not selected
