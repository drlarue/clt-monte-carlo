import numpy as np
from scipy import stats

from plotly import tools
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go

# disable for web app
#import matplotlib.pyplot as plt
#import seaborn as sns

class ExploreCLT:
    """
    Selects the parent population from which samples will be drawn.
    Example input: `lamba n: np.random.normal(size=n)`

    Arguments:
        * sampler: population distribution
        * mu: population mean
    """
    def __init__(self, sampler, mu):
        self.sampler = sampler
        self.mu = mu


    def sampling_mean(self, sample_size, sim_reps=10000):
        """
        Repeatedly draws samples of size n from the population,
        computing the t-statistic of each sample.
        Returns t-statistics (array) and sample size n as a tuple.

        Arguments:
            * sample_size: size of the samples drawn
            * sim_reps: number of times samples are drawn
        """
        t_list = np.empty(sim_reps)

        for i in range(sim_reps):
            current_sample = self.sampler(sample_size)
            t_list[i] = ((np.mean(current_sample) - self.mu) /
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


    def find_alpha(self, sorted_t_list, percentile, sample_size, desired_alpha):
        """
        Given a desired alpha, computes the actual alpha (for a two-tailed test) from the sampling
        distribution.
        """
        desired_t = stats.t.ppf(desired_alpha/2, df=sample_size-2)
        i_lower = np.searchsorted(sorted_t_list, desired_t, side='left')
        i_upper = np.searchsorted(sorted_t_list, -desired_t, side='right')

        # Another approach:
        '''
        n = np.size(t_list[t_list < desired_t]) + np.size(t_list[t_list > -desired_t])
        n /= len(t_list)
        print('Actual alpha for a two-tailed test:', n)
        '''
        return percentile[i_lower] + (1 - percentile[i_upper])


    def report(self, sampling_mean_output, desired_alpha):
        """
        Runs all the functions and outputs the following:
            * Actual test size α from the sampling distribution based on the critical values at
              the desired α (for a two-tailed test)
            * Two plots:
                1. The pdf (by KDE) of the sampling distribution of the t-statistics along with
                   the theoretical t-distribution
	        2. The normal probability plot of the sample t-statistics

        Arguments:
            * sampling_mean_output: output of the sampling_mean function
            * desired_alpha: desired α for a two-tailed test
        """
        t_list, sample_size = sampling_mean_output
        sorted_t_list, percentile = self.empirical_cdf(t_list)
        actual_alpha = self.find_alpha(sorted_t_list, percentile, sample_size, desired_alpha)

        t_grid = np.linspace(-3.5, 3.5, len(t_list))
        t_dist = stats.t.pdf(t_grid, df=sample_size-2) # theoretical t-distribution

        # Compute the Kullback-Leibler divergence
        kl_divergence = stats.entropy(stats.gaussian_kde(t_list).pdf(t_grid), t_dist)

        theo_q = stats.t.ppf(percentile, df=sample_size-2) # theoretical quantiles

        # Plotting with plotly for dash
        # left plot
        distplot = ff.create_distplot([t_list], ['Sampling distribution of standardized mean'],
                                      show_hist=False, show_rug=False)
        # Inserting a plot generated via a FigureFactory function into a subplot is a bit tricky:
        # http://nbviewer.jupyter.org/gist/empet/7eaf06b3cb5e488b129bb8df8fdb9b4b
        trace1 = distplot['data']

        for item in trace1:
            item.pop('xaxis', None)
            item.pop('yaxis', None)

        trace2 = go.Scatter(x=t_grid, y=t_dist, mode='lines',
                            line=dict(color='#d70504', dash='dot'),
                            name='Theoretical t-distribution with df={}'.format(sample_size-2))

        # right plot
        trace3 = go.Scatter(x=theo_q, y=sorted_t_list, mode='markers', showlegend=False,
                            marker=dict(size=3, color='#b5b5b5'))
        trace4 = go.Scatter(x=[-4, 4], y=[-4, 4], mode='lines', showlegend=False,
                            line=dict(width=1, color='#d70504'))

        fig = tools.make_subplots(rows=1, cols=2,
                subplot_titles=('Theoretical vs Sampling Distribution of Standardized Mean',
                                'Normal Probability Plot'))
        fig.append_trace(trace1[0], 1, 1)
        fig.append_trace(trace2, 1, 1)
        fig.append_trace(trace3, 1, 2)
        fig.append_trace(trace4, 1, 2)

        fig['layout'].update(title='Actual α: {:.4f}, KL Divergence*: {:.4f}'.
                             format(actual_alpha, kl_divergence),
                             titlefont=dict(
                                 family='Courier New, monospace',
                                 size=18,
                                 color='#0029cc'))
        fig['layout']['xaxis1'].update(range=[-3.5, 3.5], dtick=1)
        fig['layout']['yaxis1'].update(range=[0, 0.5])
        fig['layout']['legend'].update(dict(x=0.075, y=-0.3))
        fig['layout']['xaxis2'].update(title='Theoretical quantiles')
        fig['layout']['yaxis2'].update(title='Observed t-statistics')


        ''' Plotting with matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        # Left plot
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
        ax2.scatter(theo_q, sorted_t_list, c='lightblue', s=10)
        # Add x=y line for visual comparison
        sns.regplot(np.array([0, 1]), np.array([0, 1]), ax=ax2, ci=None, scatter=False,
                    line_kws={'color':'grey', 'lw':1})
        ax2.set_title('Normal Probability Plot', fontsize=15)
        ax2.set_xlabel('Theoretical quantiles')
        ax2.set_ylabel('Observed t-statistics')
        '''
        return fig
