import pandas as pd
import numpy as np
import json

from scipy.stats import norm


# Define the LogL function to calculate the log-likelihood
def LogL(theta, pi, y, X):
    N, T = y.shape
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

    for i in range(N):
        ll_store = np.array([
            np.sum(norm.logpdf(y[:, i], loc=theta[k * 2] + theta[k * 2 + 1] * X[:, 1], scale=1)) + np.log(pi[k])
            for k in range(K)
        ])
        max_ll = np.max(ll_store)
        P[i, :] = np.exp(ll_store - max_ll)
        P[i, :] /= np.sum(P[i, :])

    return P


# Define the MStep function for the Maximization step of the EM algorithm
def MStep(y, X, P):
    N, T = y.shape
    K = P.shape[1]
    theta = np.zeros(2 * K)
    pi = np.zeros(K)

    for k in range(K):
        weights = P[:, k]
        weighted_X = X * weights[:, np.newaxis]
        for i in range(N):
            weighted_y = y[:, i] * weights
            theta_k, _, _, _ = np.linalg.lstsq(weighted_X, weighted_y, rcond=None)
            theta[k * 2: (k * 2) + 2] += theta_k / N

        pi[k] = weights.mean()

    return theta, pi


def initialize_parameters(y, K):
    # Use the quantiles of y to initialize the alpha parameters for each segment
    theta = np.zeros(2 * K)
    for k in range(K):
        # For alpha (intercepts), use quantiles for initialization
        theta[2 * k] = np.quantile(y, (k + 1) / (K + 1))

        # For beta (slopes), you might start with zero or small random values
        # since we don't have prior knowledge about the distribution of slopes
        theta[2 * k + 1] = np.random.rand()

    # Initialize pi (mixing coefficients) to be equal for all components initially
    pi = np.full(K, 1.0 / K)

    return theta, pi


# Define the EM function for iterating the E and M steps
def EM(K, y, X, tol=1e-4, max_iter=100):
    # Use the new initialization function instead of random initialization
    theta, pi = initialize_parameters(y, K)
    prev_log_likelihood = None

    for _ in range(max_iter):
        P = EStep(theta, pi, y, X)
        theta, pi = MStep(y, X, P)
        log_likelihood = LogL(theta, pi, y, X)

        if prev_log_likelihood is not None and np.abs(log_likelihood - prev_log_likelihood) < tol:
            break
        prev_log_likelihood = log_likelihood

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

    # Preparing the data for the EM algorithm
    y = np.log(data.iloc[:, 1:].values)  # Log of sales
    X = np.column_stack((np.ones(data.shape[0]), np.log(data['Price'].values)))  # Adding a constant term and log prices

    N, T = y.shape  # Number of stores and time periods

    # Initialize a dictionary to store the results
    results = {'K': [], 'Theta': [], 'Pi': [], 'Log Likelihood': [], 'BIC': []}

    # Apply the Estimate function for K = 2, 3, 4 and select the best model based on BIC
    for K in [2, 3, 4]:
        theta, pi, log_likelihood = Estimate(K, y, X)
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
