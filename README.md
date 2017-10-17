# Exploring CLT via Monte Carlo Simulation

For a single sample (consisting of `n` independent observations), the Central Limit Theorem (CLT) is typically invoked to argue that the sample mean approximately follows a Normal (or t) distribution.
But exactly how good is this approximation?

To answer this question (i.e., to study how good the finite-sample approximation is), I built a [web app](https://explore-clt.herokuapp.com/) that simulates the frequentist thought experiment of repeated sampling.
This is achieved by Monte Carlo simulation, where samples of a given size are repeatedly drawn from a known parent population and the t-statistic is computed for each sample. 
With this simulated sampling distribution of the t-statistics in hand, the app evaluates the properties of the sample mean estimator and outputs the following:
* Actual test size `α` from the sampling distribution based on the critical values for the desired `α` (for a two-tailed test)
* Two plots:
	1. The PDF (from Kernel Density Estimation) of the sampling distribution of the t-statistics along with the theoretical t-distribution 
		* The Kullback-Leibler divergence from the theoretical distribution to the sampling distribution is also calculated
	2. The normal probability plot of the sample t-statistics

Play around with the app to see how good the Normal approximation is for different population distributions (4 options) and sample sizes.


## Example 

Let's consider a case where the parent population is an exponential distribution, and the sample size is 50 -- here's a screenshot of the [web app](https://explore-clt.herokuapp.com/):
![screenshot](https://user-images.githubusercontent.com/26487650/31673814-05f339e4-b315-11e7-9dde-1cd732c3a3fe.png)

Based on the critical values for a desired `α` of 0.05, the actual `α` is 0.0629, so the test has a higher false positive rate than expected.
The Normal probability plot shows that the Normal approximation is poor at n=50.

