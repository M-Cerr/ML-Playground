import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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


def apply_categorical_encoding(df, selected_columns, encoding_method, drop_option=None):
    """
    Applies categorical encoding to the specified columns in the dataset.
    
    Parameters:
    - df (pd.DataFrame): The original dataset.
    - selected_columns (list): Columns to encode.
    - encoding_method (str): The encoding method ("One-Hot Encoding").
    - drop_option (str): The option for dropping columns ("Drop first column", "Drop binary columns", "Keep all columns").
    
    Returns:
    - pd.DataFrame: The dataset with encoded categorical variables.
    """
    df_encoded = df.copy()

    if encoding_method == "One-Hot Encoding":
        encoder = OneHotEncoder(sparse_output=False, dtype=int)

        # Apply One-Hot Encoding
        encoded_data = encoder.fit_transform(df_encoded[selected_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(selected_columns))

        # Handle Drop Option
        if drop_option == "Drop first column":
            encoded_df = encoded_df.iloc[:, len(selected_columns):]  # Drop first encoding column per feature
        elif drop_option == "Drop binary columns":
            binary_cols = [col for col in encoded_df.columns if '_0' in col]
            encoded_df.drop(columns=binary_cols, inplace=True)

        # Drop original categorical columns & merge with the main dataset
        df_encoded.drop(columns=selected_columns, inplace=True)
        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

    return df_encoded
