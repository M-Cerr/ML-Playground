import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
import numpy as np
from history_manager import DatasetHistory, record_new_change


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


def apply_categorical_encoding(df, selected_columns, encoding_method, encoding_params):
    """
    Applies categorical encoding to the specified columns in the dataset.
    
    Parameters:
    - df (pd.DataFrame): The dataset being modified.
    - selected_columns (list): Columns to encode.
    - encoding_method (str): The encoding method.
    - encoding_params (dict): Additional parameters for encoding methods.
    
    Returns:
    - pd.DataFrame: The updated dataset with encoded categorical variables.
    """
    df_encoded = df.copy()

    if encoding_method == "One-Hot Encoding":
        # Determine drop behavior
        drop_strategy = None
        if encoding_params["drop_option"] == "Drop first column in all features":
            drop_strategy = "first"
        elif encoding_params["drop_option"] == "Drop first column if binary feature":
            drop_strategy = "if_binary"

        encoder = OneHotEncoder(sparse_output=False, dtype=int, drop=drop_strategy)

        # Apply One-Hot Encoding
        encoded_data = encoder.fit_transform(df_encoded[selected_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(selected_columns))

        # Drop only selected categorical columns & merge with dataset
        df_encoded.drop(columns=selected_columns, inplace=True)
        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    
    elif encoding_method == "Label Encoding":
        for col in selected_columns:
            encoder = LabelEncoder()
            try:
                df_encoded[col] = encoder.fit_transform(df_encoded[col])
            except ValueError:
                if encoding_params["handle_unknown"] == "Assign -1":
                    df_encoded[col] = df_encoded[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
                elif encoding_params["handle_unknown"] == "Ignore":
                    continue
                else:
                    raise ValueError(f"Unknown category found in {col}")

    elif encoding_method == "Ordinal Encoding":
        for col in selected_columns:
            if col in encoding_params["custom_order"]:
                custom_order = encoding_params["custom_order"][col]
                encoder = OrdinalEncoder(categories=[custom_order])
                df_encoded[col] = encoder.fit_transform(df_encoded[[col]])

    elif encoding_method == "Count Encoding":
        for col in selected_columns:
            counts = df_encoded[col].value_counts()
            if encoding_params["apply_log"]:
                counts = np.log1p(counts)  # Apply log transformation
            df_encoded[col] = df_encoded[col].map(counts)

    return df_encoded
