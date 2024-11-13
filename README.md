# Sales Mixture Model with EM Algorithm

This repository contains an implementation of the Expectation-Maximization (EM) algorithm to estimate parameters in a mixture model for weekly store-level sales data.

## 1. General Information
- **Programming Language**: Python
- **Objective**: Estimate parameters for a mixture model using the EM algorithm, without relying on pre-built mixture model libraries.
- **Libraries**: Use `pandas`, `numpy`, and `scipy` for data processing, linear algebra, and calculating density functions.

## 2. Model Overview
This model analyzes sales data at the store level for a particular product. The model assumes that sales in each store depend on the price of the product, with parameters specific to each store that vary across segments. The sales data for store *i* in week *t* is given by:

$$
\log S_{it} = \alpha_i + \beta_i \log p_t + \epsilon_{it}, \quad \epsilon_{it} \sim N(0,1)
$$

where:
- $S_{it}$: Sales in store *i* during week *t*.
- $p_t$: Price of the product, consistent across stores each week.

### Model Assumptions
- **Heterogeneous Store Parameters**: Store-specific intercept $\alpha_i$ and price elasticity $\beta_i$ vary across segments and follow a mixture distribution.
- **Segment Allocation**: Each store belongs to an unobserved cluster $C_i \in \{1, \dots, K\}$, with probabilities $\pi = (\pi_1, \dots, \pi_K)^T$ where $\sum_{c=1}^{K} \pi_c = 1$.
- **Parameter Notation**: $\theta = (\alpha[1], \beta[1], \dots, \alpha[K], \beta[K])^T$ represents the parameters for each segment.

## 3. Implementation Steps

The following functions were implemented to estimate the model parameters:

1. **Log Likelihood Function** (`LogL`): Evaluates the log-likelihood of the model.
   - **Inputs**: $\theta$, $\pi$, $y$ (log sales matrix), $X$ (constant and log prices matrix).
   - **Output**: Scalar log-likelihood value.

2. **Expectation Step (E-Step)** (`EStep`): Calculates the conditional cluster probabilities for each store.
   - **Inputs**: $\theta$, $\pi$, $y$, $X$.
   - **Output**: $N \times K$ matrix of conditional probabilities.

3. **Maximization Step (M-Step)** (`MStep`): Updates the model parameters by solving weighted least squares for each segment and calculating a closed-form solution for $\pi$.
   - **Inputs**: $y$, $X$, $W$ (conditional segment probabilities matrix).
   - **Outputs**: Updated estimates of $\theta$ and $\pi$.

4. **EM Algorithm** (`EM`): Iterates the E-step and M-step until convergence.
   - **Inputs**: $K$, $y$, $X$, tolerance, and maximum iterations.
   - **Outputs**: Final estimates of $\theta$ and $\pi$, along with the log-likelihood.

5. **Parameter Estimation** (`Estimate`): Runs the EM algorithm multiple times with different initializations and selects the solution with the highest log-likelihood.
   - **Inputs**: Number of segments $K$, data $y$, $X$, and number of initializations.
   - **Outputs**: Optimal $\theta$, $\pi$, and corresponding log-likelihood.

## 4. Application and Results

1. **Data File**: Load the provided data file (e.g., `data/609948.csv`).
2. **Run for Different Segment Numbers**: Test the model with $K = 2, 3, 4$ segments to assess the performance.
3. **Model Selection with BIC**:
   - For each $K$, report:
     - Estimated parameters ($\theta$ and $\pi$)
     - Log-likelihood
     - Bayesian Information Criterion (BIC), calculated as:

     $$
     \text{BIC} = k \log(NT) - 2 \log L
     $$

     where $k$ is the total number of free parameters.
   - Choose the model with the lowest BIC value as the best fit.

4. **Results Storage**: The results, including the parameters, log-likelihood, and BIC for each $K$, are saved in `results.json`.

---
