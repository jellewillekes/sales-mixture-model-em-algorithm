import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

# Helper functions for the regression mixture model
def e_step(X, y, theta, pi, K):
    N = len(y)
    P = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            alpha_k, beta_k = theta[k]
            P[i, k] = norm.pdf(y[i], loc=alpha_k + beta_k * X[i, 1], scale=1) * pi[k]
        P[i, :] /= np.sum(P[i, :])  # Normalize probabilities
    return P

def m_step(X, y, P, K):
    N, T = X.shape
    theta = np.zeros((K, 2))  # [alpha, beta] for each segment
    pi = np.zeros(K)
    for k in range(K):
        weights = P[:, k]
        reg = LinearRegression().fit(X[:, 1].reshape(-1, 1), y, sample_weight=weights)
        theta[k] = [reg.intercept_, reg.coef_[0]]
        pi[k] = np.mean(weights)
    return theta, pi

def log_likelihood(X, y, theta, pi, K):
    log_likelihood = 0
    for i in range(len(y)):
        ll_store = np.array([
            norm.logpdf(y[i], loc=theta[k][0] + theta[k][1] * X[i, 1], scale=1) + np.log(pi[k])
            for k in range(K)
        ])
        log_likelihood += np.log(np.sum(np.exp(ll_store)))
    return log_likelihood

def EM_regression_mixture(K, y, X, tol=1e-4, max_iter=100):
    # Initialize parameters
    theta = np.random.randn(K, 2)  # Random initialization
    pi = np.ones(K) / K  # Equal probability for each cluster
    for iteration in range(max_iter):
        P = e_step(X, y, theta, pi, K)
        theta, pi = m_step(X, y, P, K)
        current_log_likelihood = log_likelihood(X, y, theta, pi, K)
        if iteration > 0 and np.abs(previous_log_likelihood - current_log_likelihood) < tol:
            break
        previous_log_likelihood = current_log_likelihood
    return theta, pi, current_log_likelihood

# Main analysis function
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

    T, N = y.shape  # Number of stores and time periods

    # Initialize a dictionary to store the results
    results = {'K': [], 'Theta': [], 'Pi': [], 'Log Likelihood': [], 'BIC': []}

    # Apply the regression mixture model for K = 2, 3, 4 and select the best model based on BIC
    for K in [2, 3, 4]:
        theta, pi, log_likelihood = EM_regression_mixture(K, y, X)
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
    with open('results_regression_mixture.json', 'w') as json_file:
        json.dump(results, json_file)

if __name__ == "__main__":
    run_analysis()
