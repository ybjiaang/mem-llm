import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from itertools import product
import torch
import os

def get_filenames(directory_path):
    filenames = []
    # List all files in the directory
    with os.scandir(directory_path) as entries:
        for entry in entries:
            # Check if it's a file (not a directory)
            if entry.is_file():
                if 'data' in entry.name:
                    continue
                else:
                    filenames.append(directory_path + '/' + entry.name)
            else:
                filenames += get_filenames(directory_path + '/' + entry.name)
    return filenames
