# Exploring CLT via Monte Carlo Simulation

For a single sample (consisting of `n` independent observations), the Central Limit Theorem (CLT) is typically invoked to argue that the sample mean approximately follows a Normal distribution.
But exactly how good is this approximation? 

To answer this question, we can simulate the frequentist thought experiment of repeated sampling.
This can be achieved by Monte Carlo simulation, where samples are repeatedly drawn from a known parent population and the t-statistic is computed for each sample. 
With this simulated sampling distribution of the t-statistics in hand, we can evaluate the properties of the sample mean estimator.


This code provides a selection of methods to evaluate the Normal approximation for different population distributions and sample sizes.

The `ExploreCLT` class has a `report` method that outputs the following:

* Two plots:
	1. The pdf (by KDE) of the sampling distribution of the t-statistics along with the theoretical t-distribution 
		* The Kullback-Leibler divergence from the theoretical distribution to the sampling distribution is also calculated
	2. The normal probability plot of the sample t-statistics
* Actual test size `α` from the sampling distribution based on the critical values for the desired `α` (for a two-tailed test)


## Example 

Let's consider a case where the parent population is an exponential distribution, and the sample size is 50.

```python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
%matplotlib inline

from clt_mc import ExploreCLT

exponential = ExploreCLT(lambda n: np.random.exponential(size=n))
exponential.report(sample_size=50, desired_alpha=0.05)
```

This generates the following output:
```
Desired alpha for a two-tailed test: 0.05
Actual alpha for a two-tailed test: 0.0628937106289
```
![example_plots](https://user-images.githubusercontent.com/26487650/29498411-a6db72aa-85b0-11e7-8d16-aef457bbd9d1.png)

Based on the critical values for a desired `α` of 0.05, the actual `α` is 0.0629, which indicates a higher false positive rate. 
The Normal probability plot shows that Normal approximation is poor at n=50.
