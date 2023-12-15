import pandas as pd
import numpy as np
import json

from scipy.stats import norm
from scipy.special import logsumexp

"""I use sklearn for initializing parameters, I do not use sklearn for the EM algorithm itself. 
 I do this such that the EM algorithm converges faster. Else, I use IQR strategy for initializing the
 slopes of the parameters. Random initialization would simply take too long for convergence, also not stable.
 Using a linear regression on aggregated sales data for initializing beta because it provides a data-driven
 estimate of the relationship between price and sales, offering a realistic starting point for the price elasticity 
 parameter in the model."""
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans


def LogL(theta, pi, y, X):
    K = pi.shape[0]
    N, T = y.shape
    log_likelihood = 0

    for i in range(N):
        for t in range(T):
            log_likelihood_it = -np.inf
            for c in range(K):
                alpha_c, beta_c = theta[c]
                epsilon_it = y[i, t] - (alpha_c + beta_c * X[t, 1])

                # Use log probability function directly to avoid underflow
                log_phi_epsilon_it = norm.logpdf(epsilon_it, 0, 1)

                # Adding a small constant to pi[c] to avoid log(0)
                log_likelihood_it = np.logaddexp(log_likelihood_it, np.log(pi[c] + 1e-10) + log_phi_epsilon_it)

            log_likelihood += log_likelihood_it

    # Handle -inf or NaN in log likelihood
    if np.isinf(log_likelihood) or np.isnan(log_likelihood):
        log_likelihood = -np.inf

    return log_likelihood


# Define the EStep function for the Expectation step of the EM algorithm
def EStep(theta, pi, y, X):
    N, T = y.shape
    K = len(pi)
    P = np.zeros((N, K))

    for k in range(K):
        alpha_k, beta_k = theta[k]
        P[:, k] = norm.logpdf(y, loc=alpha_k + beta_k * X[:, 1], scale=1).sum(axis=1) + np.log(pi[k] + 1e-10)

    # Subtract the logsumexp to normalize
    P = np.exp(P - logsumexp(P, axis=1)[:, None])

    return P


# Define the MStep function for the Maximization step of the EM algorithm
def MStep(y, X, W):
    N, T = y.shape
    K = W.shape[1]
    theta = np.zeros((K, 2))
    pi = np.zeros(K)

    for k in range(K):
        weights = W[:, k][:, np.newaxis]
        y_weighted = np.dot(y * weights, X)

        XTX = np.dot(X.T, X)
        XTy = y_weighted.sum(axis=0)

        theta[k] = np.linalg.solve(XTX, XTy)

        # Regularize pi values
        pi[k] = weights.mean()

    # Ensure pi sums to 1
    pi = pi / pi.sum()

    return theta, pi


def initialize_parameters(y, X, K, method='quantiles'):
    # Initialize theta as a Kx2 array, where each row represents a segment with alpha and beta parameters
    theta = np.zeros((K, 2))
    pi = np.full(K, 1.0 / K)

    # Correctly aggregate log sales across all stores for each time period
    aggregated_log_sales = np.log(y.sum(axis=0))  # Log of sum of sales across all stores

    lin_reg = LinearRegression()
    lin_reg.fit(X, aggregated_log_sales)
    beta_init = lin_reg.coef_[0]  # Slope from linear regression

    if method == 'quantiles':
        for k in range(K):
            # Set alpha (intercept) as the quantile of aggregated log sales
            theta[k, 0] = np.quantile(aggregated_log_sales, (k + 1) / (K + 1))
            # Initialize beta (slope) using linear regression result
            theta[k, 1] = beta_init

    elif method == 'kmeans':
        # Reshape aggregated_log_sales for KMeans
        aggregated_log_sales_reshaped = aggregated_log_sales.reshape(-1, 1)
        kmeans = KMeans(n_clusters=K, n_init=10, random_state=0).fit(aggregated_log_sales_reshaped)
        centroids = kmeans.cluster_centers_.flatten()

        for k in range(K):
            # Set alpha (intercept) as the centroid
            theta[k, 0] = centroids[k]
            # Initialize beta (slope) using linear regression result
            theta[k, 1] = beta_init

    return theta, pi


# Define the EM function for iterating the E and M steps
def EM(K, y, X, tol=1e-4, max_iter=30):
    theta, pi = initialize_parameters(y, X, K, method='quantiles')
    prev_log_likelihood = -np.inf
    converged = False  # Track convergence

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
        # if (iteration + 1) % 10 == 0:
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

    for iter in range(n_init):
        print(f'Start of EM {iter} / {n_init}')
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

    # Log-transform the prices using np.log1p for numerical stability
    log_prices = np.log1p(prices - 1)  # Subtract 1 so np.log1p will compute the natural log of prices

    # Create an X matrix with an intercept term and the log prices
    X = np.column_stack((np.ones(len(log_prices)), log_prices))  # X should be a T x 2 matrix

    # Extract sales data, add a small constant to ensure all values are positive
    sales = data.iloc[:, 1:].values + 1e-9  # Add a small constant to avoid log(0)

    # Log-transform the sales data using np.log1p for numerical stability
    log_sales = np.log1p(sales - 1)

    # Transpose y so that each row corresponds to a store and each column to a week
    y = log_sales.T

    # Verify that T is the number of weeks and N is the number of stores
    N, T = y.shape
    assert X.shape[0] == T, "The number of weeks (rows) in X must match the number of weeks in Y"

    # Initialize a dictionary to store the results
    results = {'K': [], 'Theta': [], 'Pi': [], 'Log Likelihood': [], 'BIC': []}

    # Apply the Estimate function for K = 2, 3, 4 and select the best model based on BIC
    for K in [2, 3, 4]:
        print(f'\n Starting EM for {K} clusters:')
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
