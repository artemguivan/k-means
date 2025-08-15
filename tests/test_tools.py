import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools import get_euclidian_distance, get_dist, init_points, create_random_data


class TestTools(unittest.TestCase):
    def setUp(self):
        self.test_data = np.array([[1, 2, 3], [4, 5, 6]])
        self.vec1 = np.array([1, 2])
        self.vec2 = np.array([4, 6])
    
    def test_get_euclidian_distance_basic(self):
        result = get_euclidian_distance(self.vec1, self.vec2)
        expected = np.sqrt((1-4)**2 + (2-6)**2)
        self.assertAlmostEqual(result, expected, places=7)
    
    def test_get_euclidian_distance_same_vectors(self):
        result = get_euclidian_distance(self.vec1, self.vec1)
        self.assertEqual(result, 0.0)
    
    def test_get_euclidian_distance_different_shapes(self):
        vec3 = np.array([1, 2, 3])
        result = get_euclidian_distance(self.vec1, vec3)
        self.assertIsInstance(result, Exception)
    
    def test_get_euclidian_distance_zero_vector(self):
        zero_vec = np.array([0, 0])
        result = get_euclidian_distance(self.vec1, zero_vec)
        expected = np.sqrt(1**2 + 2**2)
        self.assertAlmostEqual(result, expected, places=7)
    
    def test_get_dist(self):
        centers = np.array([[1, 3], [2, 4]])
        sample = np.array([1, 2])
        k = 2
        
        distances = get_dist(sample, centers, k)
        
        self.assertEqual(len(distances), k)
        self.assertIsInstance(distances, list)
        
        expected_dist1 = np.sqrt((1-1)**2 + (2-2)**2)
        expected_dist2 = np.sqrt((1-3)**2 + (2-4)**2)
        
        self.assertAlmostEqual(distances[0], expected_dist1, places=7)
        self.assertAlmostEqual(distances[1], expected_dist2, places=7)
    
    def test_init_points(self):
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        k = 2
        
        result = init_points(data, k, random_seed=42)
        
        self.assertEqual(result.shape, (2, k))
        self.assertTrue(np.all(np.isin(result, data)))
    
    def test_init_points_default_k(self):
        data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        
        result = init_points(data)
        
        self.assertEqual(result.shape, (2, 3))
    
    def test_init_points_reproducible(self):
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        k = 2
        
        result1 = init_points(data, k, random_seed=42)
        result2 = init_points(data, k, random_seed=42)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_create_random_data(self):
        n_samples = 5
        
        result = create_random_data(n_samples)
        
        self.assertEqual(result.shape, (2, n_samples * 3))
        self.assertTrue(np.all(result >= 10))
        self.assertTrue(np.all(result <= 31))
    
    def test_create_random_data_structure(self):
        n_samples = 2
        
        result = create_random_data(n_samples)
        
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], n_samples * 3)


if __name__ == '__main__':
    unittest.main()