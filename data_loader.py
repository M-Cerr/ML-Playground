import pandas as pd

# Function to load a dataset
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to load all datasets
def load_sample_datasets():
    dataset_paths = {
        'Sample 1 - Food Delivery Times': 'SampleDatasets/FoodDeliveryTimes-Sample1.csv',
        'Sample 2 - Car Details': 'SampleDatasets/CarDetailsV4-Sample2.csv',
        'Sample 3 - Walmart Weekly Sales': 'SampleDatasets/WalmartSales-Sample3.csv',
    }
    
    datasets = {}
    for name, path in dataset_paths.items():
        datasets[name] = load_dataset(path)
    
    return datasets

def load_user_dataset(uploaded_file):
    """
    Load and validate a user-uploaded CSV file. Return a pandas DataFrame if valid, or None if invalid.
    Validation includes:
      - Checking it's truly a CSV (via the file extension or type, though we also rely on st.file_uploader).
      - Attempting to read with pd.read_csv.
      - Checking for 'Unnamed' columns, which typically indicate missing headers.
    """
    # Basic name check
    if not uploaded_file.name.lower().endswith('.csv'):
        print("Error: The uploaded file is not recognized as a CSV format.")
        return None

    try:
        # Try reading the CSV into a DataFrame
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print(f"Error loading uploaded CSV file: {e}")
        return None

    # Check for columns named "Unnamed..." => indicates missing or empty headers
    if any(col.startswith("Unnamed") for col in df.columns):
        print("Error: CSV has missing headers or corrupted data. 'Unnamed' columns found.")
        return None

    return df