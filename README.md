# MixtureModel-SalesAnalysis

1 General information
Use Python as programming language for the assignment below. You are not allowed to use pre-programmed packages that implement (variants of) a mixture model. It is allowed to use packages that help you process data, do basic linear algebra, calculate density functions, etc. For example, you can use pandas, numpy and a package to calculate density functions.

2 The model 
We consider a model for the sales of a particular product, measured at the store level at a weekly frequency. Denote the sales of the product in store i = 1, . . . , N in week t = 1, . . . , T by Sit . The sales are explained by the price of the product at time t, that is, pt. All stores have the same price in any given week. We specify the following heterogeneous model 
logSit =αi +βilogpt +εit,		 εit ∼N(0,1). 
The store-specific parameters are assumed to follow a mixture distribution with K segments.1 
We use Ci ∈ {1, . . . , K} to denote the (unobserved) cluster to which store i belongs. 
Denote Pr[Ci = c] = πc, c = 1,...,K with πc ≥ 0 and SUM^(K)c=1 πc = 1.  

For stores in segment c the intercept and price elasticity equal α[c] and β[c], respectively. 
Therefore αi =α[c] and βi = β[c] if Ci =c. 
Finally, denote θ=(α[1],β[1],...,α[K],β[K])′ and π=(π1,...,πK′ )′. 

3 Assignment
In this assignment you need to implement the EM algorithm to estimate the parameters of the above model. 
The different sub-questions listed below are meant to help you structure this problem. Read all questions before you begin. Your code should (at least) contain the functions specified in the questions below. In the end you need to apply your code to a specific data set (see Canvas). The exact data set that you need to use depends on your student number. For example, if your student number is 609948, you will need to use the data data609948.csv. Find the data attached in csv format.
You need to hand in your code and a short pdf file containing: - the answers to the questions 1 and 4; - the estimates/log likelihood values/BIC and conclusion for question 8; 
It is wise to test each function that you create with the provided data and with test cases you create yourself. Do not hand in these tests. 

4 Questions
	1.	Give the log-likelihood function of the above model as well as the complete-data log- likelihood function in terms of the elements of θ and π. You can use φ(x;μ,σ2) and Φ(x; μ, σ2) to denote the density and cumulative distribution function of the normal distribution. 
	2.	Program a function called LogL that evaluates the log-likelihood function of the model. This should be a function that allows you to input θ, π, y and X and outputs the (scalar) log-likelihood value, where  - y: (T × N) matrix with log sales - X(T × 2) matrix with a constant and the log prices. 
	3.	Use the formulas provide in the lecture slides and program a function called EStep to do the E-step of the EM algorithm, that is, make a function that takes as input θ, π, y and X and returns the N × K matrix of conditional cluster probabilities, where the (i, k) entry equals Pr[Ci = k|yi, X, θ, π].  … [ INSERT E STEP LECTURE SLIDE ] …
	4.	Argue why the M-step of the algorithm comes down to: (i) solving K weighted least squares problems and (ii) calculating a (closed-form) solution for π. What are the weights in this regression? 
	5.	Program a function called MStep that performs the Maximization step of the algorithm. This function should have as input y, X and W (N × K matrix of conditional segment probabilities). The function should output the new estimates of θ and π. 
	6.	Program a function called EM that iterates the E and M step until the weights do not change anymore given a chosen tolerance. Think about how to set starting values for the entire EM algorithm (you can initialize the weights or the parameters).  - Input: K, y and X - Output: estimates of θ and π 
	7.	Write a function called Estimate to perform the estimation of the model parameters for a given value of K. This function calls EM to run the actual estimation. To avoid ending up in a bad local optimum perform 10 different calls to EM and select the solution with the highest log likelihood value. 
	8.	Apply all your code to the given data set in data609948.csv. Run your code for K = 2, 3 and 4 segments. For each value of K report the estimated parameters (θ and π), log- likelihood value and BIC value. Which value of K do you prefer based on the BIC? Recall that the BIC is defined as klog(NT)−2logL where k is the total number of free parameters (in θ and π).  



