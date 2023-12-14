import pandas as pd
import numpy as np
from scipy.stats import norm

# Load the dataset
data = pd.read_csv('data/609948.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(data.info())

print("\nFirst Few Rows:")
print(data.head())

# Preparing the data for the EM algorithm
y = np.log(data.iloc[:, 1:].values)  # Log of sales
X = np.column_stack((np.ones(data.shape[0]), np.log(data['Price'].values)))  # Adding a constant term and log prices

N, T = y.shape  # Number of stores and time periods
print("\nShapes of y and X:")
print("y shape:", y.shape)
print("X shape:", X.shape)

# Define the LogL function to calculate the log-likelihood
def LogL(theta, pi, y, X):
    K = len(pi)
    alpha, beta = theta[::2], theta[1::2]

    mu = alpha[:, np.newaxis, np.newaxis] + beta[:, np.newaxis, np.newaxis] * X[:, 1, np.newaxis]
    log_pdf_values = norm.logpdf(y, loc=mu, scale=1)

    weighted_log_pdf = log_pdf_values + np.log(pi)[:, np.newaxis, np.newaxis]
    max_log_pdf = np.max(weighted_log_pdf, axis=0)
    log_likelihood = np.sum(max_log_pdf) + np.sum(np.log(np.sum(np.exp(weighted_log_pdf - max_log_pdf), axis=0)))

    return log_likelihood

# Define the EStep function for the Expectation step of the EM algorithm
def EStep(theta, pi, y, X):
    K = len(pi)
    alpha, beta = theta[::2], theta[1::2]

    mu = alpha[:, np.newaxis, np.newaxis] + beta[:, np.newaxis, np.newaxis] * X[:, 1, np.newaxis]
    log_pdf_values = norm.logpdf(y, loc=mu, scale=1)

    weighted_log_pdf = log_pdf_values + np.log(pi)[:, np.newaxis, np.newaxis]
    max_log_pdf = np.max(weighted_log_pdf, axis=0, keepdims=True)
    P = np.exp(weighted_log_pdf - max_log_pdf)
    P /= np.sum(P, axis=0, keepdims=True)

    return P

# Define the MStep function for the Maximization step of the EM algorithm
def MStep(y, X, P):
    N, T = y.shape
    K = P.shape[1]
    theta = np.zeros(2 * K)
    pi = np.zeros(K)

    for k in range(K):
        weights = P[:, k]
        W = np.diag(weights)

        # Ensure X is of shape (52, 2) for matrix multiplication
        if X.shape[0] != T:
            X = X.T

        # Building the weighted design matrix and weighted response vector
        weighted_X = np.dot(X.T * weights, X)
        weighted_y = np.dot(X.T * weights, y.T)

        # Solving the weighted least squares problem for each segment
        theta_k = np.linalg.solve(weighted_X, weighted_y)
        theta[k * 2:(k * 2) + 2] = theta_k.ravel()

        # Update pi
        pi[k] = np.mean(weights)

    return theta, pi

# Test the functions with an example K
K = 2  # Example number of segments
theta, pi = np.random.rand(2 * K), np.full(K, 1.0 / K)  # Random initialization for testing

# Test EStep and MStep functions
P = EStep(theta, pi, y, X)
theta, pi = MStep(y, X, P)
log_likelihood = LogL(theta, pi, y, X)

print("\nTest Results:")
print("Theta:", theta)
print("Pi:", pi)
print("Log Likelihood:", log_likelihood)