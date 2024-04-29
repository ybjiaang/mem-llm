import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from itertools import product
import torch
import os
from scipy.linalg import hadamard, subspace_angles

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


def angle_estimate(Mb_1, Mb_2):
    # return np.average(np.rad2deg(subspace_angles(Mb_1.T, Mb_2.T)))
    return np.rad2deg(subspace_angles(Mb_1.T, Mb_2.T))[-1]

def rank_k_approximation(matrix, k):
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    # Keep only the first k singular values and vectors
    Uk = U[:, :k]
    sk = np.diag(s[:k])
    Vk = Vt[:k, :]
    # Reconstruct the matrix using the rank-k approximation
    approx_matrix = np.dot(Uk, np.dot(sk, Vk))
    return approx_matrix