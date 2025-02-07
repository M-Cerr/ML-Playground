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

        # Define highlighting rules for missing values & type mismatches
        highlight_rules = {
            "missing_values": highlight("rgba(255, 255, 0, 0.5)", "params.value === null || params.value === ''"),
            
            # Type mismatch highlighting (only for numerical columns)
            "type_mismatches": highlight(
                "rgba(255, 0, 0, 0.5)",
                """
                function(params) {
                    let colType = params.column.colDef.type;  // Retrieve column type
                    if (colType !== "numericColumn") { return {}; }  // Ignore categorical columns
                    if (params.value === null || params.value === '') { return {}; } // Missing values should stay yellow
                    return isNaN(params.value) ? { 'backgroundColor': 'rgba(255, 0, 0, 0.5)' } : null;
                }
                """
            ),
        }

        # Apply highlighting conditions only if column type matches
        for col in df_with_index.columns:
            col_props = {}

            # Set column width
            col_props["width"] = 150 if col != "Index" else 100  # Index column smaller

            # Make cells editable for user inputs
            col_props["editable"] = True

            # Apply precision for numeric columns
            if pd.api.types.is_numeric_dtype(df_with_index[col]):
                col_props.update(PRECISION_TWO)  # Show numbers with 2 decimal places
                col_props["type"] = ["numericColumn", "customNumericFormat"]  # Mark column as numeric

            # Pin the index column
            if col == "Index":
                col_props.update(PINLEFT)
                col_props["editable"] = False #Don't let user mess with index numbers

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
        if not updated_df.equals(st.session_state['datasets'][selected_dataset_name]):  # If changes detected
                st.session_state['datasets'][selected_dataset_name] = updated_df  # Update dataset
                st.session_state["issues"] = analyze_dataset(updated_df, st.session_state['categorical_columns'][selected_dataset_name])  # Recalculate missing values & mismatches
                st.rerun()  # Refresh UI to reflect updates
        

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

        # Categorical Data Encoding Section
        if st.sidebar.checkbox("Categorical Data Encoding"):
            st.write("### Encode Categorical Data")
        
            # Retrieve categorical columns
            categorical_columns = st.session_state['categorical_columns'].get(selected_dataset_name, [])
        
            # Ensure valid columns exist in the dataset
            available_columns = [col for col in categorical_columns if col in st.session_state['datasets'][selected_dataset_name].columns]
        
            # Preserve user-selected columns in session state (Fix Multi-Select Issue)
            if "selected_columns" not in st.session_state:
                st.session_state["selected_columns"] = []
        
            # Ensure selected columns are valid & persist selection
            selected_columns = st.multiselect(
                "Select Columns to Encode",
                options=available_columns,
                default=st.session_state["selected_columns"], 
                help="Choose categorical columns to transform into numeric values."
            )
    
            # Simply update session state without forcing a rerun
            st.session_state["selected_columns"] = selected_columns
        
            encoding_method = st.selectbox(
                "Select Encoding Method",
                ["One-Hot Encoding"],
                help="Select how to encode categorical variables."
            )

            # Additional Options for One-Hot Encoding
            drop_option = None
            if encoding_method == "One-Hot Encoding":
                drop_option = st.radio(
                    "Select Column Dropping Behavior",
                    ["Drop first column in all features", "Drop first column if binary feature", "Keep all columns"],
                    index=2  # Default: Keep all columns
                )
        
            # Ensure temp dataset only appears after encoding is applied
            if "temp_encoded_dataset" not in st.session_state:
                st.session_state["temp_encoded_dataset"] = None  # Set to None instead of copying dataset
        
            # Apply Encoding Temporarily (Does Not Update Main Table)
            if st.button("Apply Encoding"):
                try:
                    temp_df = apply_categorical_encoding(
                        st.session_state['datasets'][selected_dataset_name],  # Apply on original dataset
                        selected_columns,
                        encoding_method,
                        drop_option
                    )
                    st.session_state["temp_encoded_dataset"] = temp_df  # Store temporary encoded dataset
        
                    st.success("Encoding applied! Review the changes below before confirming.")
                except Exception as e:
                    st.error(f"Error while encoding categorical data: {e}")
        
            # Show Before & After Side-by-Side Comparison
            if st.session_state["temp_encoded_dataset"] is not None:
                st.write("### Original vs. Encoded Dataset")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("#### 🔹 Original Dataset (Before Encoding)")
                    st.dataframe(st.session_state['datasets'][selected_dataset_name])

                with col2:
                    st.write("#### 🔹 Encoded Dataset (After Encoding)")
                    st.dataframe(st.session_state["temp_encoded_dataset"])

                # Only Update the Main AgGrid Table When User Confirms Changes
                if st.button("Confirm & Update Dataset"):
                    st.session_state['datasets'][selected_dataset_name] = st.session_state["temp_encoded_dataset"]
                    st.success("Main dataset updated successfully!")
                    del st.session_state["temp_encoded_dataset"]  # Clear temp dataset
                    st.rerun()
    
        else:
            st.write("⚠️ No categorical columns available for encoding.")

        st.sidebar.checkbox("Scaling")

if __name__ == "__main__":
    main()