import streamlit as st
import pandas as pd
from agstyler import draw_grid, highlight, PRECISION_TWO, PINLEFT
from dataset_analysis import analyze_dataset
from data_loader import load_user_dataset # For user uploads
from history_manager import DatasetHistory, record_new_change
from ui_history import display_history_ui

# Helper: Normalize dataset names for keys (remove spaces and lowercase)
def normalize_dataset_name(name):
    return name.replace(" ", "").lower()

# Helper: Ensure a mapping of normalized key -> display title exists in session state.
def ensure_dataset_titles():
    if "dataset_titles" not in st.session_state:
        st.session_state["dataset_titles"] = {}
    return st.session_state["dataset_titles"]

# Helper: Ensure a history object exists for the dataset using the normalized key.
def ensure_history_for_dataset(normalized_name):
    if normalized_name not in st.session_state.get("histories", {}):
        st.session_state.setdefault("histories", {})[normalized_name] = DatasetHistory(max_size=5)

def display_dataset_selection_and_analysis():
    """
    Handles dataset selection and user setup.
    Returns:
      - selected_dataset_name: The normalized key for internal use.
      - display_title: The original title (for UI display).
      - df: The selected dataset (DataFrame).
      - updated_df: The dataset after AgGrid modifications.
      - formatter: AgGrid formatting dictionary.
    """
    st.title("ML Playground")
    st.write("Welcome to the ML Playground Tool :)")
    
    # FILE UPLOADER FOR USER DATASETS 
    st.subheader("Upload Your Own CSV (Optional)")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], 
                                     help="Upload a CSV file to add a new dataset.")
    if uploaded_file is not None:
        # Attempt to load & validate
        df_new = load_user_dataset(uploaded_file)
        if df_new is None:
            st.error("⚠️ Failed to load your CSV. Please ensure it has valid headers and isn't corrupted.")
        else:
            # Use a display title that is the original name; then normalize it for keys.
            display_title = f"Uploaded – {uploaded_file.name}"
            normalized = normalize_dataset_name(display_title)
            # Save the dataset using the normalized key.
            st.session_state.setdefault('datasets', {})[normalized] = df_new
            # Also create or update the mapping for display titles.
            titles = ensure_dataset_titles()
            titles[normalized] = display_title
            # Initialize categorical columns for this dataset.
            if 'categorical_columns' not in st.session_state:
                st.session_state['categorical_columns'] = {}
            if normalized not in st.session_state['categorical_columns']:
                st.session_state['categorical_columns'][normalized] = []
            st.success(f"✅ Uploaded and added dataset: **{display_title}**. You can select it below.")
    
    # Build a dictionary mapping display titles to normalized keys.
    titles = ensure_dataset_titles()
    # If no titles are present (e.g. only sample datasets exist), we use the keys in st.session_state['datasets']
    if not titles:
        for key in st.session_state.get('datasets', {}).keys():
            titles[key] = key  # Use normalized key as display title if no custom title exists
    
    # Create a mapping for the selectbox: display_title -> normalized key.
    option_dict = {display: norm for norm, display in titles.items()}
    # Selectbox shows display titles.
    selected_display = st.selectbox("Choose a dataset to view", options=list(option_dict.keys()))
    # Retrieve the normalized key.
    selected_dataset_name = option_dict[selected_display]
    
    # Save the chosen display title for later UI (if needed).
    display_title = selected_display

    if not selected_dataset_name:
        return None, None, None, None, {} # Prevents errors when no dataset is selected

    
    # Retrieve the dataset using the normalized key.
    df = st.session_state['datasets'][selected_dataset_name]
    st.write(f"Selected Dataset: {display_title}")

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
        ensure_history_for_dataset(selected_dataset_name) # Create the history for the confirmed table (re-do if Categorical columns are re-chosen later)
        st.success(f"Updated categorical columns: {', '.join(selected_categorical_columns)}")
        st.session_state[f"{selected_dataset_name}_done"] = True

    # Proceed with dataset analysis if user confirmed selection
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
            "type_mismatches": highlight(
                "rgba(255, 0, 0, 0.5)",
                """
                function(params) {
                    let colType = params.column.colDef.type;
                    if (colType !== "numericColumn") { return {}; }
                    if (params.value === null || params.value === '') { return {}; }
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
                col_props["editable"] = False  # Don't let user mess with index numbers

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
            - **🔴 Red** → Type Mismatch  
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
            record_new_change(
                selected_dataset_name, 
                updated_df,
                "Manually updated a cell in the dataset"
            )
            st.session_state["issues"] = analyze_dataset(updated_df, st.session_state['categorical_columns'][selected_dataset_name])  # Recalculate missing values & mismatches
            st.rerun()  # Refresh UI to reflect updates

        return selected_dataset_name, df, updated_df, formatter  # Return for further processing in app.py
    
    return selected_dataset_name, df, None, {}
