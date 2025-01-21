import json
import os

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder to handle numpy data types.
    """
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)  # Convert np.float32 to native Python float
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to a Python list
        return super(NumpyEncoder, self).default(obj)


def save_to_json(output_folder, output_file, data):
    """
    Save data to a JSON file, creating the folder if it doesn't exist.

    Parameters:
    - output_folder (str): Path to the folder where the file will be saved.
    - output_file (str): Name of the JSON file (without extension).
    - data (dict): Data to be saved in JSON format.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists
    with open(f"./{output_folder}/{output_file}.json", "w") as json_file:
        json.dump(data, json_file, cls=NumpyEncoder, indent=2)  # Save data with custom encoder and formatting


def load_from_json(filename):
    """
    Load data from a JSON file.

    Parameters:
    - filename (str): Path to the JSON file.

    Returns:
    - dict: Parsed data from the JSON file.
    """
    with open(filename, "r") as json_file:
        return json.load(json_file)  # Load and return the JSON data