import numpy as np

def get_euclidian_distance(
        vec1: np.array,
        vec2: np.array
        ) -> np.float64:
    
    vec1_flatten = vec1.flatten()
    vec2_flatten = vec2.flatten()

    if vec1_flatten.shape != vec2_flatten.shape:
        return Exception("sizes of vectors are diff")

    squared_diff = (vec1_flatten - vec2_flatten) ** 2

    sum_of_squares = squared_diff.sum()

    return np.sqrt(sum_of_squares)

def get_dist(
    sample: np.ndarray,
    centers: np.ndarray,
    k: int
    ) -> list[float]:
    distances = [get_euclidian_distance(sample, centers[:, j]) for j in range(k)]
    return distances

def init_points(
    data: np.ndarray,
    k: int = 3,
    random_seed: int = 927
    ) -> np.ndarray: 
    n_points = data.shape[-1]
    
    np.random.seed(random_seed)   
    random_indices = np.random.choice(n_points, k, replace=False)
    cluster_centers = data[:, random_indices]
    return cluster_centers

def create_random_data(
    n_samples: int
    ) -> np.ndarray:
    row1 = [val + np.random.random() for _ in range(n_samples) for val in range(10, 30 + 1, 10)]
    row2 = [val + np.random.random() for _ in range(n_samples) for val in range(10, 30 + 1, 10)]

    data = np.array([row1, row2])

    return data
