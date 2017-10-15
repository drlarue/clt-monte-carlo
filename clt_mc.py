import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class ExploreCLT:
    """
    Selects the parent population from which samples will be drawn.
    Example input: `lamba n: np.random.normal(size=n)`

    Arguments:
        * sampler: population distribution
    """
    def __init__(self, sampler):
        self.sampler = sampler


    def sampling_mean(self, sample_size, sim_reps=10000):
        """
        Repeatedly draws samples of size n from the population,
        computing the t-statistic of each sample.
        Returns t-statistics (array) and sample size n as a tuple.

        Arguments:
            * sample_size: size of the samples drawn
            * sim_reps: number of times samples are drawn
        """
        population = self.sampler(100000)
        t_list = np.empty(sim_reps)

        for i in range(sim_reps):
            current_sample = self.sampler(sample_size)
            t_list[i] = ((np.mean(current_sample) - np.mean(population)) /
                         (np.std(current_sample, ddof=1) / np.sqrt(sample_size)))

        return t_list, sample_size


    def empirical_cdf(self, raw_data):
        """
        Returns sorted data and the corresponding percentiles as a tuple.
        """
        data = np.sort(raw_data)

        #percentile = np.ones_like(data).cumsum() / (len(data) + 1)
        # [1/(N+1), ..., N/(N+1)]

        percentile = (np.arange(len(data)) + 0.5)/ len(data)
        # [0.5/N, ..., (N-0.5)/N]
        # https://stackoverflow.com/a/11692365/588071

        return data, percentile


    def plot_dist_npp(self, sampling_mean_output):
        """
        Generates the following two plots:
            1. Theoretical vs sampling distribution of standardized mean
            2. Normal probability plot
        """
        t_list, sample_size = sampling_mean_output

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        # Left plot
        t_grid = np.linspace(-3.5, 3.5, len(t_list))
        t_dist = stats.t.pdf(t_grid, df=sample_size-2)
        ax1.plot(t_grid, t_dist, c='r',
                 label='Theoretical t-distribution with df={}'.format(sample_size-2))
        sns.distplot(t_list, norm_hist=True, hist=False, kde=True, kde_kws={'color':'orange'},
                     label='Sampling distribution of standardized mean', ax=ax1)
        ax1.legend(loc=9, fontsize=12)
        ax1.set_title('Theoretical vs Sampling Distribution of Standardized Mean', fontsize=15)
        ax1.set_ylim(0, 0.5)
        ax1.set_xlim(-3.5, 3.5)
        ax1.text(1.5, 0.35, 'KL Divergence:')
        ax1.text(1.5, 0.325,
                 np.round(stats.entropy(stats.gaussian_kde(t_list).pdf(t_grid), t_dist), 4))

        # Right plot
        sorted_t_list, percentile = self.empirical_cdf(t_list)
        theo_q = stats.t.ppf(percentile, df=sample_size-2)
        ax2.scatter(theo_q, sorted_t_list, c='lightblue', s=10)
        # Add x=y line for visual comparison
        sns.regplot(np.array([0, 1]), np.array([0, 1]), ax=ax2, ci=None, scatter=False,
                    line_kws={'color':'grey', 'lw':1})
        ax2.set_title('Normal Probability Plot', fontsize=15)
        ax2.set_xlabel('Theoretical quantiles')
        ax2.set_ylabel('Observed t-statistics')


    def find_alpha(self, sampling_mean_output, desired_alpha):
        """
        Given a desired alpha, computes the actual alpha (for a two-tailed test) from the sampling
        distribution and prints it.
       """
        t_list, sample_size = sampling_mean_output
        sorted_t_list, percentile = self.empirical_cdf(t_list)

        desired_t = stats.t.ppf(desired_alpha/2, df=sample_size-2)
        i_lower = np.searchsorted(sorted_t_list, desired_t, side='left')
        i_upper = np.searchsorted(sorted_t_list, -desired_t, side='right')

        print('Desired alpha for a two-tailed test:', desired_alpha)
        print('Actual alpha for a two-tailed test:', percentile[i_lower] + (1 - percentile[i_upper]))

        # Another approach:
        '''
        n = np.size(t_list[t_list < desired_t]) + np.size(t_list[t_list > -desired_t])
        n /= len(t_list)
        print('Actual alpha for a two-tailed test:', n)
        '''


    def report(self, sample_size, desired_alpha, sim_reps=10000):
        """
        Runs all the functions and outputs the following:
            * Actual test size α from the sampling distribution based on the critical values at
              the desired α (for a two-tailed test)
            * Two plots:
                1. The pdf (by KDE) of the sampling distribution of the t-statistics along with
                   the theoretical t-distribution
	        2. The normal probability plot of the sample t-statistics

        Arguments:
            * sample_size: size of the samples drawn
            * desired_alpha: desired α for a two-tailed test
            * sim_reps: number of times samples are drawn
        """
        sampling_distribution = self.sampling_mean(sample_size, sim_reps)
        self.find_alpha(sampling_distribution, desired_alpha)
        self.plot_dist_npp(sampling_distribution)
