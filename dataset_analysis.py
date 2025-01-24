import pandas as pd

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