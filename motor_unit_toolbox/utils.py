import numpy as np

def firings_to_binary(firings: list, signal_length: int) -> np.ndarray:
    """Convert firings list to binary matrix representation

    Args:
        firings (list): List of firings arrays
        signal_length (int): Length of the original signal

    Returns:
        np.ndarray: Binary matrix of shape (signal_length, n_units)
    """
    n_units = len(firings)
    binary_matrix = np.zeros((signal_length, n_units), dtype=bool)

    for unit_idx, unit_firings in enumerate(firings):
        valid_firings = unit_firings[unit_firings < signal_length]
        binary_matrix[valid_firings, unit_idx] = True

    return binary_matrix

def binary_to_firings(binary_matrix: np.ndarray) -> list:
    """Convert binary matrix representation to firings list

    Args:
        binary_matrix (np.ndarray): Binary matrix of shape (signal_length, n_units)

    Returns:
        list: List of firings arrays
    """
    firings = []
    for unit_idx in range(binary_matrix.shape[1]):
        unit_firings = np.where(binary_matrix[:, unit_idx])[0]
        firings.append(unit_firings)
    return firings