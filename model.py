import numpy as np
from src.tools import init_points, get_dist, create_random_data

def cluster_using_kmeans(
        data: np.ndarray,
        k: int = 3, 
        max_iters: int = 100,
        atol: float = 1e-9
        ) -> tuple[list[list[np.ndarray]], np.ndarray]:
    
    if not isinstance(data, np.ndarray):
          data = np.array(data)

    n_points = data.shape[-1]
    cluster_centers = init_points(data, k)

    
    for iteration in range(max_iters):
        clusters = [[] for _ in range(k)]  
        cluster_assignments = {}

        for i in range(n_points):
            sample = data[:, i]
            distances = get_dist(sample, cluster_centers, k)      
            
            closest_cluster = np.argmin(distances) # 0,1,2...,k
            clusters[closest_cluster].append(sample)
            cluster_assignments[i] = closest_cluster

        for idx, cluster in enumerate(clusters):
            if len(cluster) == 0:

                clusters[idx] = init_points(data, k).T
           
        centroids = []
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(cluster, axis=0)
            centroids.append(new_centroid)
            

        if np.allclose(cluster_centers, np.array(centroids).T, atol=atol):
            print(f"end of process: {iteration + 1}")
            break
   
        cluster_centers = np.array(centroids).T

    return clusters, cluster_centers

if __name__ == "__main__":

    data = create_random_data(10)

    clusters, cluster_centers = cluster_using_kmeans(data)
    print(f"shapes: {[np.array(cluster).shape for cluster in clusters]}")

