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
        'Sample 1': 'datasets/sample1.csv',
        'Sample 2': 'datasets/sample2.csv',
        'Sample 3': 'datasets/sample3.csv',
    }
    
    datasets = {}
    for name, path in dataset_paths.items():
        datasets[name] = load_dataset(path)
    
    return datasets
