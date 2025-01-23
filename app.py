import streamlit as st
from data_loader import load_sample_datasets
import pandas as pd 
import numpy as np
 
# Function to analyze the dataset
def analyze_dataset(df, categorical_columns):
    issues = {
        "missing_values": {},
        "type_mismatches": {}
    }
    
    # Check for missing values
    missing = df.isnull()
    if missing.any().any():
        for col in df.columns:
            if missing[col].any():
                issues["missing_values"][col] = list(df.index[missing[col]])

    # Check for type mismatches
    for col in df.columns:
        if col in categorical_columns:
            continue  # Skip type checks for user-defined categorical columns

        # Handle numeric columns
        mismatches = None
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if values are numeric or null
            mismatches = df[~df[col].apply(lambda x: isinstance(x, (int, float)) or pd.isnull(x))]
        else:
            # If not numeric dtype, check if all values can be coerced to numeric
            try:
                pd.to_numeric(df[col], errors="raise")
            except ValueError:
                mismatches = df[~df[col].apply(lambda x: isinstance(x, (int, float)) or pd.isnull(x))]

        # Record mismatches
        if mismatches is not None and not mismatches.empty:
            issues["type_mismatches"][col] = list(mismatches.index)
    
    return issues

# Function to style the dataframe with see-through highlights
def highlight_issues(df, issues):
    def highlight_cell(val, row_idx, col_name):
        if col_name in issues["missing_values"] and row_idx in issues["missing_values"][col_name]:
            return "background-color: rgba(255, 255, 0, 0.3)"  # Light yellow
        if col_name in issues["type_mismatches"] and row_idx in issues["type_mismatches"][col_name]:
            return "background-color: rgba(255, 0, 0, 0.3)"  # Light red
        return ""

    # Apply style
    return df.style.apply(
        lambda col: [
            highlight_cell(val, idx, col.name) for idx, val in enumerate(col)
        ],
        axis=0,
    )

def handle_missing_values(df, column, method, custom_value=None, is_categorical=False):
    """
    Handles missing values in a specified column based on the selected method.

    Parameters:
        df (pd.DataFrame): The dataset.
        column (str): The column to process.
        method (str): The method to handle missing values ('Mean', 'Median', 'Mode', 'Custom Value').
        custom_value (str/float/int, optional): The value to use for 'Custom Value' method.
        is_categorical (bool): Whether the column is categorical.

    Returns:
        pd.DataFrame: The updated DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the dataset.")

    # Numeric column handling
    if not is_categorical:
        if method == "Mean":
            df[column].fillna(df[column].mean(), inplace=True)
        elif method == "Median":
            df[column].fillna(df[column].median(), inplace=True)
        elif method == "Mode":
            df[column].fillna(df[column].mode()[0], inplace=True)
        elif method == "Custom Value":
            if custom_value is None:
                raise ValueError("Custom value must be provided for the 'Custom Value' method.")
            df[column].fillna(float(custom_value), inplace=True)
        else:
            raise ValueError(f"Unsupported method '{method}' for numeric columns.")

    # Categorical column handling
    else:
        if method == "Mode":
            df[column].fillna(df[column].mode()[0], inplace=True)
        elif method == "Custom Value":
            if custom_value is None:
                raise ValueError("Custom value must be provided for the 'Custom Value' method.")
            df[column].fillna(custom_value, inplace=True)
        else:
            raise ValueError(f"Unsupported method '{method}' for categorical columns.")

    return df


# Streamlit app
def main():
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

        header_only = pd.DataFrame(columns=df.columns)

        st.write("### Step 2: Mark Categorical Columns")
        st.write("Click the checkboxes below to mark columns as categorical. When done, click 'Proceed'.")
        column_selections = st.columns(len(df.columns))

        # Adjustable layout for checkboxes
        max_cols_per_row = 4  # Adjust this number to control horizontal spacing
        col_names = df.columns.tolist()
        num_cols = len(col_names)
        rows = (num_cols // max_cols_per_row) + (1 if num_cols % max_cols_per_row != 0 else 0)
        
        for row in range(rows):
            with st.container():
                row_cols = st.columns(min(max_cols_per_row, num_cols - row * max_cols_per_row))
                for idx, col in enumerate(row_cols):
                    col_idx = row * max_cols_per_row + idx
                    if col_idx < num_cols:
                        column_name = col_names[col_idx]
                        is_categorical = column_name in st.session_state['categorical_columns'][selected_dataset_name]
                        if col.checkbox(f"{column_name}", value=is_categorical, key=f"cat_col_{column_name}"):
                            if column_name not in st.session_state['categorical_columns'][selected_dataset_name]:
                                st.session_state['categorical_columns'][selected_dataset_name].append(column_name)
                        else:
                            if column_name in st.session_state['categorical_columns'][selected_dataset_name]:
                                st.session_state['categorical_columns'][selected_dataset_name].remove(column_name)

        if st.button("Proceed"):
            st.session_state[f"{selected_dataset_name}_done"] = True

    # Step 3: Analyze and display the dataset
    if st.session_state.get(f"{selected_dataset_name}_done"):
        st.write("### Step 3: Dataset Analysis")
        issues = analyze_dataset(df, st.session_state['categorical_columns'][selected_dataset_name])
        styled_df = highlight_issues(df, issues)
        st.dataframe(styled_df, use_container_width=True, height=500)

        # Display checklist summary with expanders
        st.sidebar.header("Dataset Analysis Summary")

        # Missing values summary
        # Add clickable cell links for auto-scroll (HTML + JavaScript injection)
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

    # Step 4: Missing Value Replacement Section
    if st.sidebar.checkbox("Missing Value Replacement"):
        st.write("### Step 4: Handle Missing Values")
        
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
        updated_df = st.session_state['datasets'][selected_dataset_name]
        st.dataframe(updated_df, use_container_width=True, height=500)
        
    
    st.sidebar.checkbox("Scaling")

if __name__ == "__main__":
    main()