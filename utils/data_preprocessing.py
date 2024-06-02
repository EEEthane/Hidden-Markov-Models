import numpy as np

def preprocess_data(file_path):
    """
    Reads a file and preprocesses the data into a format suitable for training/testing HMM.
    
    Args:
    - file_path (str): Path to the data file.

    Returns:
    - observations (list of int): List of observations as integers.
    """
    with open(file_path, 'r') as file:
        data = file.read().strip().split()
    observations = [int(obs) for obs in data]
    return observations

def load_data(file_path):
    """
    Load and preprocess data from a file, ensuring the data is in the correct numerical format.

    Args:
    - file_path (str): Path to the data file.

    Returns:
    - data (list): List of data points.
    """
    with open(file_path, 'r') as file:
        data = file.read().strip().split()
    return data

def convert_to_integers(data):
    """
    Convert list of observations to integers.

    Args:
    - data (list): List of observations.

    Returns:
    - (list of int): List of observations as integers.
    """
    unique_items = sorted(set(data))
    item_to_int = {item: i for i, item in enumerate(unique_items)}
    return [item_to_int[item] for item in data]

def preprocess_data_generic(file_path):
    """
    Preprocess data by loading it and converting observations to integers.

    Args:
    - file_path (str): Path to the data file.

    Returns:
    - (list of int): List of observations as integers.
    """
    data = load_data(file_path)
    return convert_to_integers(data)
