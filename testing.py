import unittest
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from main import LogL, EStep, MStep, EM, Estimate


class TestEMFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the dataset just once for all tests
        cls.data = pd.read_csv('data/609948.csv')
        cls.y = np.log(cls.data.iloc[:, 1:].values)
        cls.X = np.column_stack((np.ones(cls.data.shape[0]), np.log(cls.data['Price'].values)))
        cls.N, cls.T = cls.y.shape

        # Fit a Gaussian Mixture Model to obtain expected values
        cls.gmm = GaussianMixture(n_components=2, random_state=0)
        cls.gmm.fit(np.hstack((cls.y.T, cls.X[:, 1:])))  # Assuming the second column is the log price

        cls.expected_pi = cls.gmm.weights_
        cls.expected_means = cls.gmm.means_[:, 1:]  # Assuming the first column is the intercept
        cls.expected_precisions = cls.gmm.precisions_[:, 1:, 1:]  # Assuming diagonal covariance

    def test_LogL(self):
        # Test LogL function with expected values from GaussianMixture
        expected_log_likelihood = self.gmm.lower_bound_ * self.T
        theta = self.expected_means.flatten()
        pi = self.expected_pi
        result = LogL(theta, pi, self.y, self.X)
        self.assertAlmostEqual(result, expected_log_likelihood, places=5)

    def test_EStep(self):
        # Test EStep function with expected values from GaussianMixture
        expected_probabilities = self.gmm.predict_proba(np.hstack((self.y.T, self.X[:, 1:])))
        theta = self.expected_means.flatten()
        pi = self.expected_pi
        result = EStep(theta, pi, self.y, self.X)
        np.testing.assert_array_almost_equal(result, expected_probabilities, decimal=5)

    def test_MStep(self):
        # Generate some probabilities for testing MStep
        probabilities = self.gmm.predict_proba(np.hstack((self.y.T, self.X[:, 1:])))
        # Perform MStep using these probabilities
        theta, pi = MStep(self.y, self.X, probabilities)
        # Since MStep outputs new theta and pi, compare them to the GaussianMixture's means and weights
        expected_theta = self.expected_means.flatten()
        expected_pi = self.expected_pi
        np.testing.assert_array_almost_equal(theta, expected_theta, decimal=5)
        np.testing.assert_array_almost_equal(pi, expected_pi, decimal=5)

    def test_EM(self):
        # For EM, since it involves random initialization, set a seed before calling your EM function
        np.random.seed(0)
        # Call your EM function
        K = 2  # Number of components
        theta, pi, log_likelihood = EM(K, self.y, self.X)
        # Compare the results with the GaussianMixture's parameters
        expected_theta = self.expected_means.flatten()
        expected_pi = self.expected_pi
        np.testing.assert_array_almost_equal(theta, expected_theta, decimal=5)
        np.testing.assert_array_almost_equal(pi, expected_pi, decimal=5)

    def test_Estimate(self):
        # Since Estimate involves multiple initializations and picks the best one,
        # you may just want to check if the result is reasonable by checking if it's close to the GaussianMixture's result
        K = 2  # Number of components
        best_theta, best_pi, best_log_likelihood = Estimate(K, self.y, self.X)
        # The best log likelihood should be close to the one from GaussianMixture
        expected_log_likelihood = self.gmm.lower_bound_ * self.T
        self.assertAlmostEqual(best_log_likelihood, expected_log_likelihood, places=5)


if __name__ == '__main__':
    unittest.main()
