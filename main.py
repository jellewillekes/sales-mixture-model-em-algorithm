import pandas as pd
import numpy as np
from scipy.stats import norm

data = pd.read_csv('data/609948.csv')

# Display basic information about the dataset
data_info = data.info()
data_head = data.head()

# Calculate descriptive statistics for the data
descriptive_stats = data.describe()

# Calculating correlation between price and sales for each store
correlation_data = data.corr().loc['Price', 'Sales1':'Sales300']

# Print some data to get a better understanding of the dataset
print(data_info)
print(data_head)
print(descriptive_stats)
print(correlation_data.head())

# Correct the shapes of y and X
y = np.log(data.iloc[:, 1:].values)  # Log of sales (52 weeks x 300 stores), no transpose needed
X = np.column_stack((np.ones(data.shape[0]), np.log(data['Price'].values)))  # Constant and log prices (52 weeks x 2)

N = 300
T = 52


# Step 2: Define the LogL function.
def LogL(theta, pi, y, X):
    K = len(pi)  # Number of segments
    N, T = y.shape
    log_likelihood = 0

    for i in range(N):
        log_likelihood_store = np.array([
            np.sum(norm.logpdf(y[:, i], loc=theta[k * 2] + theta[k * 2 + 1] * X[:, 1], scale=1)) + np.log(pi[k])
            for k in range(K)
        ])
        max_log_likelihood = np.max(log_likelihood_store)
        log_likelihood += max_log_likelihood + np.log(np.sum(np.exp(log_likelihood_store - max_log_likelihood)))

    return log_likelihood


# Step 3: Define the EStep function.
def EStep(theta, pi, y, X):
    N, T = y.shape
    K = len(pi)  # Number of segments
    P = np.zeros((N, K))  # Conditional cluster probabilities

    for i in range(N):
        log_likelihood_store = np.array([
            np.sum(norm.logpdf(y[:, i], loc=theta[k * 2] + theta[k * 2 + 1] * X[:, 1], scale=1)) + np.log(pi[k])
            for k in range(K)
        ])
        max_log_likelihood = np.max(log_likelihood_store)
        P[i, :] = np.exp(log_likelihood_store - max_log_likelihood)  # Use the log-sum-exp trick
        P[i, :] /= np.sum(P[i, :])  # Normalize

    return P


# Step 4: Define the MStep function.
def MStep(y, X, P):
    K = P.shape[1]
    N, T = y.shape
    theta = np.zeros(2 * K)
    pi = np.zeros(K)

    for k in range(K):
        weights = P[:, k]  # Weights for each observation
        # Weighted least squares for each segment k
        weighted_X = X * weights[:, np.newaxis]  # This is correct, we're weighting each row of X by the weights
        # Since y is (T x N), we need to weight each store's sales data by the weights
        # and solve for each store separately.
        for i in range(N):
            weighted_y = y[:, i] * weights  # Weighted sales data for the i-th store
            # Solve weighted least squares problem
            theta_k, _, _, _ = np.linalg.lstsq(weighted_X, weighted_y, rcond=None)
            # Since we're solving for each store, we need to collect the results accordingly
            theta[k * 2: (k * 2) + 2] += theta_k / N  # Average the results across stores

        # Update the mixture probabilities
        pi[k] = weights.mean()

    return theta, pi


# Step 5: Define the EM function.
def EM(K, y, X, tol=1e-4, max_iter=100):
    N, T = y.shape
    theta = np.random.rand(2 * K)  # Initialize parameters randomly
    pi = np.full(K, 1.0 / K)  # Start with equal probabilities

    for _ in range(max_iter):
        P = EStep(theta, pi, y, X)  # E-step
        theta, pi = MStep(y, X, P)  # M-step
        log_likelihood = LogL(theta, pi, y, X)  # Calculate log-likelihood
        if np.abs(log_likelihood - LogL(theta, pi, y, X)) < tol:
            break  # Convergence check

    return theta, pi, log_likelihood


# Step 6: Define the Estimate function.
def Estimate(K, y, X, n_init=10):
    best_log_likelihood = -np.inf
    best_theta = None
    best_pi = None

    for _ in range(n_init):
        theta, pi, log_likelihood = EM(K, y, X)
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_theta = theta
            best_pi = pi

    return best_theta, best_pi, best_log_likelihood


# Step 7: Apply the Estimate function for K = 2, 3, 4 and select the best model based on BIC.
best_bic = np.inf
best_K = None
best_results = None
for K in [2, 3, 4]:
    theta, pi, log_likelihood = Estimate(K, y, X)
    num_params = 2 * K + (K - 1)  # Number of parameters: 2 for each segment plus one less than K for pi
    bic = num_params * np.log(N * T) - 2 * log_likelihood
    if bic < best_bic:
        best_bic = bic
        best_K = K
        best_results = (theta, pi, log_likelihood)

# Print the best results
print(f"Best number of segments based on BIC: {best_K}")
print(f"Best Theta: {best_results[0]}")
print(f"Best Pi: {best_results[1]}")
print(f"Best Log Likelihood: {best_results[2]}")
print(f"Best BIC: {best_bic}")
