import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import cluster_using_kmeans
from src.tools import create_random_data


class TestModel(unittest.TestCase):
    def setUp(self):
        self.simple_data = np.array([[1, 2, 8, 9], [1, 2, 8, 9]])
        self.test_data = create_random_data(3)
    
    def test_cluster_using_kmeans_basic(self):
        clusters, cluster_centers = cluster_using_kmeans(self.simple_data, k=2, max_iters=10)
        
        self.assertEqual(len(clusters), 2)
        self.assertEqual(cluster_centers.shape[0], 2)
        self.assertEqual(cluster_centers.shape[1], 2)
    
    def test_cluster_using_kmeans_with_list_input(self):
        data_list = [[1, 2, 8, 9], [1, 2, 8, 9]]
        clusters, cluster_centers = cluster_using_kmeans(data_list, k=2, max_iters=10)
        
        self.assertEqual(len(clusters), 2)
        self.assertIsInstance(cluster_centers, np.ndarray)
    
    def test_cluster_using_kmeans_single_cluster(self):
        clusters, cluster_centers = cluster_using_kmeans(self.simple_data, k=1, max_iters=10)
        
        self.assertEqual(len(clusters), 1)
        self.assertEqual(cluster_centers.shape[1], 1)
    
    def test_cluster_using_kmeans_default_parameters(self):
        clusters, cluster_centers = cluster_using_kmeans(self.test_data)
        
        self.assertEqual(len(clusters), 3)
        self.assertEqual(cluster_centers.shape[1], 3)
    
    def test_cluster_using_kmeans_convergence(self):
        data = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
        clusters, cluster_centers = cluster_using_kmeans(data, k=2, max_iters=100, atol=1e-6)
        
        self.assertEqual(len(clusters), 2)
        for cluster in clusters:
            self.assertGreater(len(cluster), 0)
    
    def test_cluster_using_kmeans_early_convergence(self):
        data = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
        clusters, cluster_centers = cluster_using_kmeans(data, k=1, max_iters=100)
        
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 4)
    
    def test_cluster_using_kmeans_empty_cluster_handling(self):
        data = np.array([[1, 1, 10, 10, 20], [1, 1, 10, 10, 20]])
        clusters, cluster_centers = cluster_using_kmeans(data, k=4, max_iters=50)
        
        self.assertEqual(len(clusters), 4)
        self.assertEqual(cluster_centers.shape[1], 4)
    
    def test_cluster_using_kmeans_return_types(self):
        clusters, cluster_centers = cluster_using_kmeans(self.simple_data, k=2)
        
        self.assertIsInstance(clusters, list)
        self.assertIsInstance(cluster_centers, np.ndarray)
        
        for cluster in clusters:
            self.assertIsInstance(cluster, list)
    
    def test_cluster_using_kmeans_cluster_assignments(self):
        data = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
        clusters, cluster_centers = cluster_using_kmeans(data, k=2, max_iters=100)
        
        total_points = sum(len(cluster) for cluster in clusters)
        self.assertEqual(total_points, data.shape[1])
    
    def test_cluster_using_kmeans_tolerance(self):
        data = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        
        clusters_strict, _ = cluster_using_kmeans(data, k=2, atol=1e-10)
        clusters_loose, _ = cluster_using_kmeans(data, k=2, atol=1e-1)
        
        self.assertEqual(len(clusters_strict), 2)
        self.assertEqual(len(clusters_loose), 2)


if __name__ == '__main__':
    unittest.main()