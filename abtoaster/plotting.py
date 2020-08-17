
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


CLRS_LT = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
CLRS_DK = ['#2980b9', '#c0392b', '#27ae60', '#f1c40f']


def __stats(a, b, ax):
    """
    Function that compute statistics for a plot's points.
    """
    stats = """
    Pearson\'s r = %.2f
    MAPE = %.2f%%
    Euclidean distance = %.4f
    """ % (
        sp.stats.pearsonr(a, b)[0],
        np.mean(np.abs(np.where(a != 0, (a - b) / a, 0)))*100,
        np.linalg.norm(a-b)
    )

    ax.text(
        0,
        0.91,
        stats,
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes
    )

    return stats


def ppplot(a, b, distr_names=['First', 'Second'], figsize=(8, 8),
           num_points=100, splitted=False, log_transform=False):
    """
    P-P plot - plots two cumulative distribution functions against each other.

    In statistics, a P–P plot [1] (probability–probability plot or percent–percent
    plot) is a probability plot for assessing how closely two data sets agree,
    which plots the two cumulative distribution functions against each other.
    P-P plots are vastly used to evaluate the skewness of a distribution.

    Parameters
    ----------
        a, b : list
            List, numpy array or pandas series with values
        distr_names : list of str
            Names for two distribuions that will be plotted on axes
        num_points : int
            Number of points on a plot - detalization parameter
        splitted : bool
            Split (or not) plot into four plots for each quartile
        log_transform : bool
            Perform (or not) log transform on observed data

    Returns
    -------

    [matplotlib.axes._axes.Axes]
        List of matplotlib Axes that contains p-p plot

    References
    ---------

    [1] P-P plot - https://en.wikipedia.org/wiki/P-P_plot

    """
    def plot(a_p, b_p, ax, q):
        ax.scatter(a_p, b_p, s=30, c=CLRS_DK[0])
        ax.plot(q, q, c=CLRS_DK[1], alpha=0.75)
        return ax

    a = np.array(a)
    b = np.array(b)

    if log_transform:
        a = np.log(a)
        b = np.log(b)

    a_p, b_p = [], []
    lower_bound = min(a.min(), b.min())
    upper_bound = max(a.max(), b.max())
    z_space = np.linspace(lower_bound, upper_bound, num_points)

    for z in z_space:
        a_p.append((a <= z).mean())
        b_p.append((b <= z).mean())

    if not splitted:
        fig, ax = plt.subplots(figsize=figsize)
        ax = plot(a_p, b_p, ax, q=[0, 1])
        ax.set_title('P-P plot')
        ax.set_xlabel("%s cumulative distribution" % distr_names[0])
        ax.set_ylabel("%s cumulative distribution" % distr_names[1])

        __stats(np.array(a_p), np.array(b_p), ax)
        return [ax]
    else:
        fig, axs = plt.subplots(2, 2, figsize=figsize)

        quartiles = [[.0, .25], [.25, .50], [.50, .75], [.75, 1]]

        for q, ax in zip(quartiles, np.ravel(axs)):
            ax = plot(
                list(
                    filter(lambda z: 1 if z >= q[0] and z < q[1] else 0, a_p)
                ),
                list(
                    filter(lambda z: 1 if z >= q[0] and z < q[1] else 0, b_p)
                    ),
                ax,
                q
            )

        plt.suptitle('P-P plots for each quartiles')
        return axs


def qqplot(a, b, distr_names=['First', 'Second'],
           num_points=100, fitted_line=False,
           splitted=False, figsize=(8, 8), log_transform=False):
    """
    Q-Q plot - plots quantiles of two distributions against each other.
    Useful for comparing shape of distributions.

    In statistics, a Q–Q plot[1] ("Q" stands for quantile) is a probability
    plot, which is a graphical method for comparing two probability
    distributions by plotting their quantiles against each other.
    First, the set of intervals for the quantiles is chosen. A point (x, y) on
    the plot corresponds to one of the quantiles of the second distribution
    (y-coordinate) plotted against the same quantile of the first distribution
    (x-coordinate). Thus the line is a parametric curve with the parameter
    which is the (number of the) interval for the quantile.

    Parameters
    ----------
        a, b : list
            List, numpy array or pandas series with values
        distr_names : list of str
            Names for two distribuions that will be plotted on axes
        num_points : int
            Number of points on a plot - detalization parameter
        splitted : bool
            Split (or not) plot into four plots for each quartile
        log_transform : bool
            Perform (or not) log transform on observed data
        fitted_line : bool
            Perform (or not) linear approximation for comparing line

    Returns
    -------

    [matplotlib.axes._axes.Axes]
        List of matplotlib Axes that contains p-p plot

    References
    ---------

    [1] Q-Q plot - https://en.wikipedia.org/wiki/Q-Q_plot

    """
    def plot(q_a, q_b, line, ax):
        ax.scatter(q_a, q_b, s=30, c=CLRS_DK[0])
        ax.plot(q_a, line, c=CLRS_DK[1], alpha=0.75)
        return ax

    a = np.array(a)
    b = np.array(b)

    if log_transform:
        a = np.log(a)
        b = np.log(b)

    q_a = np.array([np.percentile(a, q/(num_points/100))
                    for q in range(0, num_points + 1)])
    q_b = np.array([np.percentile(b, q/(num_points/100))
                    for q in range(0, num_points + 1)])

    if fitted_line:
        coeff = np.polyfit(q_a, q_b, 1)
        line = np.array(q_a) * coeff[0] + coeff[1]
    else:
        line = q_a

    if not splitted:
        fig, ax = plt.subplots(figsize=figsize)
        ax = plot(q_a, q_b, line, ax)
        ax.set_title('Q-Q plot')
        ax.set_xlabel("%s distribution's quantiles" % distr_names[0])
        ax.set_ylabel("%s distribution's quantiles" % distr_names[1])
        __stats(np.array(q_a), np.array(q_b), ax)
        return [ax]

    else:
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        quartiles = [[.0, .25], [.25, .50], [.50, .75], [.75, 1]]

        for q, ax in zip(quartiles, np.ravel(axs)):
            start = int(q[0] * len(q_a))
            stop = int(q[1] * len(q_a))
            ax = plot(q_a[start:stop], q_b[start:stop], line[start:stop], ax)
        plt.suptitle('Q-Q plots for each quartiles')
        return axs

def abplot(a, b, alpha=0.05, hist_kwds={}):
    """
    Plot two histograms of A and B side-by-side.

    Parameters
    ----------
    a, b : {list, ndarray}
        Observed data arrays
    alpha : float
        Plot (1 - alpha/2) and (alpha/2) percentiles
    hist_kwds : dict
        Matplotlib's hist keywords for customization

    Returns
    -------
    ax : matplotlib.axes
    """
    a, b = np.array(a), np.array(b)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(a, 50, label='A group', color='#3498db', alpha=0.5, **hist_kwds)
    ax.hist(b, 50, label='B group', color='#e74c3c', alpha=0.5, **hist_kwds)

    ax.axvline(np.percentile(a, (alpha/2)*100), color='#2980b9', alpha=0.5, linestyle='--')
    ax.axvline(a.mean(), color='#2980b9')
    ax.axvline(np.percentile(a, (1 - alpha/2)*100), color='#2980b9', alpha=0.5, linestyle='--')

    ax.axvline(np.percentile(b, (alpha/2)*100), color='#c0392b', alpha=0.5, linestyle='--')
    ax.axvline(b.mean(), color='#c0392b')
    ax.axvline(np.percentile(b, (1 - alpha/2)*100), color='#c0392b', alpha=0.5, linestyle='--')

    ax.set_title('A-B histogram comparsion plot')
    ax.legend()

    return ax