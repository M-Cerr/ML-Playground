import pandas as pd

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


def apply_categorical_encoding(df, selected_columns, encoding_method):
    """
    Applies categorical encoding to the specified columns in the dataset.
    
    Parameters:
    - df (pd.DataFrame): The original dataset.
    - selected_columns (list): Columns to encode.
    - encoding_method (str): The encoding method ("One-Hot Encoding").
    
    Returns:
    - pd.DataFrame: The dataset with encoded categorical variables.
    """
    df_encoded = df.copy()

    if encoding_method == "One-Hot Encoding":
        df_encoded = pd.get_dummies(df_encoded, columns=selected_columns, drop_first=True)

    return df_encoded
