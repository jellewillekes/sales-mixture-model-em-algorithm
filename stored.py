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


def LogL(theta, pi, y, X):
    """
    Evaluate the log-likelihood function of the model. Takes params theta, pi, y and x, with:

    Params:
    - theta: A 1D array containing the parameters α[c] and β[c] for each segment c.
    - pi: A 1D array containing the probabilities πc for each segment c.
    - y: A 2D array (T x N matrix) with log sales.
    - X: A 2D array (T x 2 matrix) with a constant and the log prices.

    Returns:
    - The scalar log-likelihood value.
    """

    K = len(pi)  # Number of segments
    N, T = y.shape
    log_likelihood = 0

    # Iterate over all stores and time periods
    for i in range(N):
        for t in range(T):
            # Calculate the log likelihood for each segment and store-time combination
            likelihood_segment = np.array([
                norm.logpdf(y[t, i], loc=theta[k * 2] + theta[k * 2 + 1] * X[t, 1], scale=1) + np.log(pi[k])
                for k in range(K)
            ])

            # Sum over segments to get the total likelihood for this store-time combination
            log_likelihood += np.log(np.sum(np.exp(likelihood_segment)))

    return log_likelihood

# Example usage (with dummy values for theta, pi, y, and X)
# These values need to be defined based on the specific model and data
# theta = np.array([alpha1, beta1, alpha2, beta2, ...])  # Example theta
# pi = np.array([pi1, pi2, ...])  # Example pi
# y = np.log(data.iloc[:, 1:].to_numpy().T)  # Log of sales (T x N)
# X = np.column_stack((np.ones(len(data)), np.log(data['Price'])))  # Log of prices with a constant term (T x 2)

# log_likelihood = LogL(theta, pi, y, X)
# print(log_likelihood)


def EStep(theta, pi, y, X):
    """
    Perform the E-step of the EM algorithm.

    Parameters:
    - theta: A 1D array containing the parameters α[c] and β[c] for each segment c.
    - pi: A 1D array containing the probabilities πc for each segment c.
    - y: A 2D array (T x N matrix) with log sales.
    - X: A 2D array (T x 2 matrix) with a constant and the log prices.

    Returns:
    - An N x K matrix of conditional cluster probabilities.
    """
    N, T = y.shape
    K = len(pi)  # Number of segments

    # Initialize the matrix for conditional cluster probabilities
    P = np.zeros((N, K))

    # Calculate the probabilities
    for i in range(N):
        for k in range(K):
            # Extract α and β for segment k
            alpha_k = theta[k * 2]
            beta_k = theta[k * 2 + 1]

            # Calculate the log-likelihood for segment k and store i
            log_likelihood = np.sum(norm.logpdf(y[:, i], loc=alpha_k + beta_k * X[:, 1], scale=1))

            # Calculate the conditional cluster probability for segment k and store i
            P[i, k] = np.log(pi[k]) + log_likelihood

        # Convert log probabilities to probabilities and normalize
        P[i, :] = np.exp(P[i, :] - np.max(P[i, :]))  # Subtract max for numerical stability
        P[i, :] /= np.sum(P[i, :])  # Normalize so that the probabilities sum to 1

    return P

# The actual values for theta and pi would need to be defined based on the specific problem and data.
# Example usage (with dummy values for theta, pi, y, and X):
# theta = np.array([...])  # Array of α and β for each segment
# pi = np.array([...])  # Array of segment probabilities π
# y = ...  # Matrix of log sales (T x N)
# X = ...  # Matrix with a constant and log prices (T x 2)

# Now we can call the function to perform the E-step
# P = EStep(theta, pi, y, X)
# print(P)  # This will print the N x K matrix of conditional cluster probabilities

