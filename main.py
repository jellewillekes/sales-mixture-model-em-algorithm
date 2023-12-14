import pandas as pd
import numpy as np
import json

from scipy.stats import norm
from scipy.special import logsumexp
from sklearn.cluster import KMeans


# Define the LogL function to calculate the log-likelihood
def LogL(theta, pi, y, X):
    T, N = y.shape
    K = len(pi)
    log_likelihood = 0

    for i in range(N):
        ll_store = np.array([
            np.sum(norm.logpdf(y[:, i], loc=theta[k * 2] + theta[k * 2 + 1] * X[:, 1], scale=1)) + np.log(pi[k])
            for k in range(K)
        ])
        max_ll = np.max(ll_store)
        log_likelihood += max_ll + np.log(np.sum(np.exp(ll_store - max_ll)))

    return log_likelihood


# Define the EStep function for the Expectation step of the EM algorithm
def EStep(theta, pi, y, X):
    N, T = y.shape
    K = len(pi)
    P = np.zeros((N, K))

    for k in range(K):
        P[:, k] = norm.logpdf(y, loc=theta[k * 2] + theta[k * 2 + 1] * X[:, 1], scale=1).sum(axis=1) + np.log(pi[k])

    P = np.exp(P - logsumexp(P, axis=1)[:, None])

    return P


# Define the MStep function for the Maximization step of the EM algorithm
def MStep(y, X, P):
    N, T = y.shape  # N: stores, T: weeks
    K = P.shape[1]
    theta = np.zeros(2 * K)
    pi = np.zeros(K)

    for k in range(K):
        weights = P[:, k]
        X_weighted = X * weights[None, :]  # Broadcasting weights across the weeks for each store

        # Weighted regression for each segment
        XTX = X_weighted @ X_weighted.T  # X'WX where W is the diagonal matrix of weights
        XTy = X_weighted @ y[k, :]  # X'Wy

        theta_k = np.linalg.solve(XTX, XTy)  # Solve for theta (alpha, beta)

        theta[k * 2: (k * 2) + 2] = theta_k
        pi[k] = weights.mean()

    return theta, pi


def initialize_parameters(y, K, method='quantiles'):
    theta = np.zeros(2 * K)
    pi = np.full(K, 1.0 / K)

    if method == 'quantiles':
        for k in range(K):
            theta[2 * k] = np.quantile(y, (k + 1) / (K + 1))
            theta[2 * k + 1] = np.random.rand()

    elif method == 'kmeans':
        # Reshape y for KMeans
        y_reshaped = y.reshape(-1, 1)
        kmeans = KMeans(n_clusters=K, n_init=10, random_state=0).fit(y_reshaped)
        centroids = kmeans.cluster_centers_.flatten()

        for k in range(K):
            theta[2 * k] = centroids[k]
            theta[2 * k + 1] = np.random.rand()  # or some other method to initialize slopes

    return theta, pi


# Define the EM function for iterating the E and M steps
def EM(K, y, X, tol=1e-5, max_iter=100):
    theta, pi = initialize_parameters(y, K, method='kmeans')
    prev_log_likelihood = -np.inf
    converged = False  # Flag to track convergence

    for iteration in range(max_iter):
        P = EStep(theta, pi, y, X)
        theta, pi = MStep(y, X, P)
        log_likelihood = LogL(theta, pi, y, X)

        # Check for convergence
        if np.abs(log_likelihood - prev_log_likelihood) < tol:
            converged = True
            print(f"Converged at iteration {iteration + 1}, Log Likelihood: {log_likelihood}")
            break
        prev_log_likelihood = log_likelihood

        # Print every 10th iteration
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}, Log Likelihood: {log_likelihood}")

    # Final summary
    if not converged:
        print("Warning: Maximum iterations reached without convergence.")
    print(f"Final Log Likelihood: {log_likelihood}")

    return theta, pi, log_likelihood


# Define the Estimate function to find the best parameters
def Estimate(K, y, X, n_init=10):
    best_log_likelihood = -np.inf
    best_theta, best_pi = None, None

    for _ in range(n_init):
        theta, pi, log_likelihood = EM(K, y, X)
        if log_likelihood > best_log_likelihood:
            best_log_likelihood, best_theta, best_pi = log_likelihood, theta, pi

    return best_theta, best_pi, best_log_likelihood


def run_analysis():
    # Load the dataset
    data = pd.read_csv('data/609948.csv')

    # Display basic information about the dataset
    print("Dataset Information:")
    print(data.info())

    print("\nFirst Few Rows:")
    print(data.head())

    # Check for zeros or negative values in the sales data (excluding the price column)
    if (data.iloc[:, 1:] <= 0).any().any():
        raise ValueError("Sales data contains zeros or negative values, which are not suitable for log transformation.")

    # Extract the price series
    prices = data['Price'].values

    # Confirm that all prices are positive before log transformation
    if (prices <= 0).any():
        raise ValueError("Price data contains zeros or negative values, which are not suitable for log transformation.")

    # Log-transform the prices
    log_prices = np.log(prices)

    # Create an X matrix with an intercept term and the log prices
    X = np.column_stack((np.ones(len(log_prices)), log_prices))  # X should be a T x 2 matrix

    # Extract and log-transform the sales data
    sales = data.iloc[:, 1:].values
    log_sales = np.log(sales)

    # Transpose y so that each row corresponds to a store and each column to a week
    y = log_sales.T

    # Confirm that T is the number of weeks and N is the number of stores
    N, T = y.shape
    assert X.shape[0] == T, "The number of weeks (rows) in X must match the number of weeks in Y"

    # Initialize a dictionary to store the results
    results = {'K': [], 'Theta': [], 'Pi': [], 'Log Likelihood': [], 'BIC': []}

    # Apply the Estimate function for K = 2, 3, 4 and select the best model based on BIC
    for K in [2, 3, 4]:
        theta, pi, log_likelihood = Estimate(K, y, X, n_init=10)
        num_params = 2 * K + K - 1
        bic = num_params * np.log(N * T) - 2 * log_likelihood

        # Store the results for each K
        results['K'].append(K)
        results['Theta'].append(theta.tolist())
        results['Pi'].append(pi.tolist())
        results['Log Likelihood'].append(log_likelihood)
        results['BIC'].append(bic)

    # Print the results for each K
    for k in range(len(results['K'])):
        print(f"\nNumber of segments (K): {results['K'][k]}")
        print(f"Theta: {results['Theta'][k]}")
        print(f"Pi: {results['Pi'][k]}")
        print(f"Log Likelihood: {results['Log Likelihood'][k]}")
        print(f"BIC: {results['BIC'][k]}")

    # Save the dictionary as a JSON file
    with open('results.json', 'w') as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    run_analysis()
