from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, ColumnsAutoSizeMode
import streamlit as st
from agstyler import draw_grid, highlight, PRECISION_TWO, PINLEFT
from data_loader import load_sample_datasets
from dataset_analysis import analyze_dataset
from dataset_analysis import highlight_issues_aggrid
from preprocessing import handle_missing_values
import pandas as pd
import numpy as np
 

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

    # Step 1: Select dataset
    st.title("ML Playground")
    st.write("Welcome to the ML Playground Tool :)")
    selected_dataset_name = st.selectbox("Choose a dataset to view", options=list(st.session_state['datasets'].keys()))

    # Step 2: Display header row and let the user mark categorical columns
    if selected_dataset_name:
        df = st.session_state['datasets'][selected_dataset_name]
        st.write(f"Selected Dataset: {selected_dataset_name}")

        # Initialize categorical columns for this dataset if not already set
        if selected_dataset_name not in st.session_state['categorical_columns']:
            st.session_state['categorical_columns'][selected_dataset_name] = []

        # Multi-select dropdown for categorical columns
        selected_categorical_columns = st.multiselect(
            "Select Categorical Columns",
            options=df.columns.tolist(),
            default=st.session_state['categorical_columns'][selected_dataset_name],
            help="Choose columns that should be treated as categorical."
        )

        # Add an "Update" button to save changes
        if st.button("Update Categorical Columns"):
            st.session_state['categorical_columns'][selected_dataset_name] = selected_categorical_columns
            st.success(f"Updated categorical columns: {', '.join(selected_categorical_columns)}")
            st.session_state[f"{selected_dataset_name}_done"] = True

    # Step 3: Analyze and highlight issues in the dataset
    if st.session_state.get(f"{selected_dataset_name}_done"):
        st.write("### Step 3: Dataset Analysis")
        
        # Analyze dataset for issues
        issues = analyze_dataset(df, st.session_state['categorical_columns'][selected_dataset_name])

        # Add index column to the DataFrame for display
        df_with_index = df.reset_index()
        df_with_index.rename(columns={"index": "Index"}, inplace=True)

        # Define formatting options for AgGrid
        formatter = {}

        # for col in df_with_index.columns:
        #     col_props = {}

        #     # Set column width
        #     col_props["width"] = 150 if col != "Index" else 100  # Index column smaller

        #     # Apply precision for numeric columns
        #     if pd.api.types.is_numeric_dtype(df_with_index[col]):
        #         col_props.update(PRECISION_TWO)  # Show numbers with 2 decimal places

        #     # Pin the index column
        #     if col == "Index":
        #         col_props.update(PINLEFT)

        #     formatter[col] = (col, col_props)

        # Define highlighting rules for missing values & type mismatches
        highlight_rules = {
            "missing_values": highlight("rgba(255, 255, 0, 0.5)", "params.value === null || params.value === ''"),
            
            # Type mismatch highlighting (only for numerical columns)
            "type_mismatches": highlight(
                "rgba(255, 0, 0, 0.5)",
                """
                function(params) {
                    let colType = params.column.colDef.type;  // Retrieve column type
                    if (colType !== "numericColumn") { return null; }  // Ignore categorical columns
                    if (params.value === null || params.value === '') { return null; }  // Missing values should stay yellow
                    return isNaN(params.value) ? { 'backgroundColor': 'rgba(255, 0, 0, 0.5)' } : null;
                }
                """
            ),
        }

        # # Apply highlighting conditions to appropriate columns
        # for col in df_with_index.columns:
        #     if col in issues["missing_values"]:
        #         formatter[col][1]["cellStyle"] = highlight_rules["missing_values"]
        #     if col in issues["type_mismatches"]:
        #         formatter[col][1]["cellStyle"] = highlight_rules["type_mismatches"]

        # Apply highlighting conditions only if column type matches
        for col in df_with_index.columns:
            col_props = {}

            # Set column width
            col_props["width"] = 150 if col != "Index" else 100  # Index column smaller

            # Apply precision for numeric columns
            if pd.api.types.is_numeric_dtype(df_with_index[col]):
                col_props.update(PRECISION_TWO)  # Show numbers with 2 decimal places
                col_props["type"] = ["numericColumn", "customNumericFormat"]  # Mark column as numeric

            # Pin the index column
            if col == "Index":
                col_props.update(PINLEFT)

            # Apply highlight for missing values
            if col in issues["missing_values"]:
                col_props["cellStyle"] = highlight_rules["missing_values"]

            # Apply highlight for type mismatches (but only for numeric columns)
            if col in issues["type_mismatches"]:
                col_props["cellStyle"] = highlight_rules["type_mismatches"]

            formatter[col] = (col, col_props)

        # Add a floating legend popover near the table
        with st.popover("❓ Help: Legend"):
            st.write(""" 
            - **🟡 Yellow** → Missing Value  
            - **🔴 Red ** → Type Mismatch  
            - **⚪ Light Grey** → Index Column  
            """)

        # Display AgGrid table using `agstyler`
        grid_response = draw_grid(
            df_with_index,
            formatter=formatter,
            fit_columns=True,  # Ensure better column sizing
            theme="streamlit",
            max_height=700,  # Keep table height large
        )
        
        # Save updates made in the table back to the session state
        updated_df = pd.DataFrame(grid_response["data"])
        updated_df.set_index("Index", inplace=True)  # Restore the original index
        st.session_state['datasets'][selected_dataset_name] = updated_df

        # Display checklist summary with expanders
        st.sidebar.header("Dataset Analysis Summary")

        # Missing values summary
        with st.sidebar.expander("⚠️ Missing Values Found" if issues["missing_values"] else "✅ No missing values detected"):
            if issues["missing_values"]:
                for col, rows in issues["missing_values"].items():
                    st.write(f"`{col}`: Rows {rows}")

        # Type mismatches summary
        with st.sidebar.expander("⚠️ Type Mismatches Found" if issues["type_mismatches"] else "✅ No type mismatches detected"):
            if issues["type_mismatches"]:
                for col, rows in issues["type_mismatches"].items():
                    st.write(f"`{col}`: Rows {rows}")

        st.sidebar.write("### Categorical Columns")
        if st.session_state['categorical_columns'][selected_dataset_name]:
            for col in st.session_state['categorical_columns'][selected_dataset_name]:
                st.sidebar.write(f"- `{col}`")
        else:
            st.sidebar.write("None selected.")

        st.sidebar.title("Preprocessing Options")

        # Missing Value Replacement Section
        if st.sidebar.checkbox("Missing Value Replacement"):
            st.write("### Handle Missing Values")

            # Iterate over columns with missing values
            for col in issues["missing_values"]:
                st.write(f"#### Column: `{col}`")

                # Determine if the column is categorical
                is_categorical = col in st.session_state['categorical_columns'][selected_dataset_name]
                col_type = "Categorical" if is_categorical else "Numeric"

                st.write(f"Column Type: **{col_type}**")

                # Select method options based on column type
                if is_categorical:
                    method_options = ["Mode", "Custom Value"]
                else:
                    method_options = ["Mean", "Median", "Mode", "Custom Value"]

                method = st.selectbox(
                    f"Select method for `{col}`",
                    method_options,
                    key=f"missing_method_{col}"
                )

                # For custom value option, add an input box
                custom_value = None
                if method == "Custom Value":
                    custom_value = st.text_input(
                        f"Enter custom value for `{col}`",
                        key=f"custom_value_{col}"
                    )

                # Apply the selected method when user clicks the button
                if st.button(f"Apply to `{col}`"):
                    try:
                        st.session_state['datasets'][selected_dataset_name] = handle_missing_values(
                            st.session_state['datasets'][selected_dataset_name],
                            col,
                            method,
                            custom_value,
                            is_categorical
                        )
                        st.success(f"Missing values in `{col}` handled successfully!")
                    except Exception as e:
                        st.error(f"Error while handling `{col}`: {e}")

            # Display the updated dataset
            st.write("### Updated Dataset")
            st.dataframe(st.session_state['datasets'][selected_dataset_name], use_container_width=True, height=500)

        st.sidebar.checkbox("Scaling")

if __name__ == "__main__":
    main()