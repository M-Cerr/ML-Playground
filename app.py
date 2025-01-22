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
        if pd.api.types.is_numeric_dtype(df[col]):
            mismatches = df[~df[col].apply(lambda x: isinstance(x, (int, float)) or pd.isnull(x))]
            if not mismatches.empty:
                issues["type_mismatches"][col] = list(mismatches.index)
    
    return issues

# Function to style the dataframe
def highlight_issues(df, issues):
    def highlight_cell(val, row_idx, col_name):
        if col_name in issues["missing_values"] and row_idx in issues["missing_values"][col_name]:
            return "background-color: yellow"
        if col_name in issues["type_mismatches"] and row_idx in issues["type_mismatches"][col_name]:
            return "background-color: red"
        return ""

    return df.style.apply(
        lambda col: [
            highlight_cell(val, idx, col.name) for idx, val in enumerate(col)
        ],
        axis=0,
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

    # Display checklist summary
    st.sidebar.header("Dataset Analysis Summary")
    if issues["missing_values"]:
        st.sidebar.write("⚠️ **Missing Values Found**")
        for col, rows in issues["missing_values"].items():
            st.sidebar.write(f"- Column `{col}`: Rows {rows}")
    else:
        st.sidebar.write("✅ No missing values detected.")
    
    if issues["type_mismatches"]:
        st.sidebar.write("⚠️ **Type Mismatches Found**")
        for col, rows in issues["type_mismatches"].items():
            st.sidebar.write(f"- Column `{col}`: Rows {rows}")
    else:
        st.sidebar.write("✅ No type mismatches detected.")
    
    st.sidebar.write("### Categorical Columns")
    if st.session_state['categorical_columns'][selected_dataset_name]:
        for col in st.session_state['categorical_columns'][selected_dataset_name]:
            st.sidebar.write(f"- `{col}`")
    else:
        st.sidebar.write("None selected.")

st.sidebar.title("Preprocessing Options")
st.sidebar.checkbox("Missing Value Replacement")
st.sidebar.checkbox("Scaling")
